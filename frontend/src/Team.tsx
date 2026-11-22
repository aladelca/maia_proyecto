import { useState } from "react";
import andresPhoto from "./assets/andres-bautista.png";
import janerPhoto from "./assets/janer-buila.png";

function Portrait({ src, name, initials }: { src?: string; name: string; initials: string }) {
  const [failed, setFailed] = useState(false);
  return <div className="team-portrait">{src && !failed ? <img src={src} alt={name} onError={() => setFailed(true)} /> : <span aria-label={`Perfil de ${name}`}>{initials}</span>}</div>;
}

export function Team() {
  return <section className="team-section" aria-labelledby="team-title">
    <p className="eyebrow">EL EQUIPO</p>
    <h2 id="team-title">Sobre nosotros</h2>
    <p className="team-intro">Tres profesionales que unen sus conocimientos en este proyecto de grado de la Maestría en Inteligencia Artificial de la Universidad de los Andes.</p>
    <div className="team-grid">
      <article className="team-card">
        <div className="team-cover" />
        <Portrait src="https://avatars.githubusercontent.com/u/83436724?s=256&v=4" name="Carlos Adrián Alarcón Delgado" initials="CA" />
        <div className="team-body">
          <h3>Carlos Adrián Alarcón Delgado</h3>
          <p className="team-position">Data Scientist · OODARIS AI</p>
          <p>Experiencia en analítica y aprendizaje de máquina. Master of Management Analytics por McGill University.</p>
          <a className="team-email" href="mailto:c.alarcond@uniandes.edu.co">c.alarcond@uniandes.edu.co</a>
        </div>
      </article>
      <article className="team-card">
        <div className="team-cover" />
        <Portrait src={andresPhoto} name="Andrés Enrique Bautista Pabón" initials="AB" />
        <div className="team-body">
          <h3>Andrés Enrique Bautista Pabón</h3>
          <p className="team-position">Senior Specialist · Banco Davivienda</p>
          <p>Ingeniero electrónico y estudiante de la Maestría en Inteligencia Artificial de la Universidad de los Andes.</p>
          <a className="team-email" href="mailto:a.bautistap@uniandes.edu.co">a.bautistap@uniandes.edu.co</a>
        </div>
      </article>
      <article className="team-card">
        <div className="team-cover" />
        <Portrait src={janerPhoto} name="John Janer Buila Ferrin" initials="JB" />
        <div className="team-body">
          <h3>John Janer Buila Ferrin</h3>
          <p className="team-position">Data Engineering · Palmas de Tumaco</p>
          <p>Ingeniero industrial y estudiante de la Maestría en Inteligencia Artificial (MAIA) de la Universidad de los Andes.</p>
          <a className="team-email" href="mailto:j.buila@uniandes.edu.co">j.buila@uniandes.edu.co</a>
        </div>
      </article>
    </div>
  </section>;
}
