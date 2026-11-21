import { request as httpsRequest } from "node:https";
import { defineConfig, loadEnv } from "vite";

// Development only: production still receives config.json from Terraform/CI.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "MAIA_DEV_");
  const target = env.MAIA_DEV_API;
  const storageHost = env.MAIA_DEV_STORAGE_HOST;
  return {
    plugins: [{
      name: "local-service-config",
      configureServer(server) {
        if (!target) return;
        server.middlewares.use(async (req, res, next) => {
          const localUrl = new URL(req.url ?? "/", "http://localhost");
          if (localUrl.pathname === "/dev-upload") {
            let asset: URL;
            try { asset = new URL(localUrl.searchParams.get("url") ?? ""); }
            catch { res.statusCode = 400; res.end("Invalid upload URL"); return; }
            const contentType = req.headers["content-type"] ?? "";
            const limit = 129 * 1024 ** 2;
            if (req.method !== "POST" || !storageHost || asset.protocol !== "https:" || asset.hostname !== storageHost || asset.port || asset.username || asset.password || asset.pathname !== "/" || asset.search || !contentType.startsWith("multipart/form-data;") || Number(req.headers["content-length"] ?? 0) > limit) {
              res.statusCode = 403; res.end("Upload not allowed"); return;
            }
            // Stream the original signed form unchanged. S3 validates its policy.
            const upstream = httpsRequest(asset, { method: "POST", headers: {
              "content-type": contentType,
              ...(req.headers["content-length"] ? { "content-length": req.headers["content-length"] } : {}),
            } }, response => {
              res.statusCode = response.statusCode ?? 502;
              res.setHeader("Cache-Control", "no-store");
              const version = response.headers["x-amz-version-id"];
              if (version) res.setHeader("x-amz-version-id", version);
              res.setHeader("Content-Type", "text/plain");
              response.pipe(res);
            });
            upstream.setTimeout(120000, () => upstream.destroy());
            upstream.on("error", () => { if (!res.headersSent) res.statusCode = 502; res.end("Upload unavailable"); });
            let bytes = 0;
            req.on("data", chunk => { bytes += chunk.length; if (bytes > limit) { req.unpipe(upstream); upstream.destroy(); if (!res.headersSent) res.statusCode = 413; res.end("Upload too large"); } });
            req.on("aborted", () => upstream.destroy());
            req.pipe(upstream);
            return;
          }
          if (localUrl.pathname === "/dev-results") {
            // Only signed GET downloads from this project's configured bucket.
            // Never forward cookies, guest tokens or Origin to storage.
            let asset: URL;
            try { asset = new URL(localUrl.searchParams.get("url") ?? ""); }
            catch { res.statusCode = 400; res.end("Invalid result URL"); return; }
            if (req.method !== "GET" || !storageHost || asset.protocol !== "https:" || asset.hostname !== storageHost || asset.port || asset.username || asset.password || !/\.(json|csv)$/.test(asset.pathname) || !asset.searchParams.has("X-Amz-Signature")) {
              res.statusCode = 403; res.end("Result URL not allowed"); return;
            }
            try {
              const upstream = await fetch(asset, { redirect: "manual", signal: AbortSignal.timeout(30000) });
              if (!upstream.ok) { res.statusCode = upstream.status >= 400 ? upstream.status : 502; res.end("Result download failed"); return; }
              const content = await upstream.arrayBuffer();
              res.setHeader("Content-Type", upstream.headers.get("content-type") ?? "application/octet-stream");
              res.setHeader("Cache-Control", "no-store");
              res.setHeader("X-Content-Type-Options", "nosniff");
              if (asset.pathname.endsWith(".csv")) res.setHeader("Content-Disposition", 'attachment; filename="probabilidades.csv"');
              res.end(Buffer.from(content));
            } catch { res.statusCode = 502; res.end("Result download unavailable"); }
            return;
          }
          if (req.url?.split("?")[0] !== "/config.json") return next();
          res.setHeader("Content-Type", "application/json");
          res.setHeader("Cache-Control", "no-store");
          res.end(JSON.stringify({ api: "/dev-api", ...(storageHost ? { resultProxy: "/dev-results", uploadProxy: "/dev-upload", storageHost } : {}) }));
        });
      },
    }],
    server: {
      proxy: target ? {
        "/dev-api/public": {
          target,
          changeOrigin: true,
          rewrite: path => path.replace(/^\/dev-api/, ""),
        },
      } : undefined,
    },
  };
});
