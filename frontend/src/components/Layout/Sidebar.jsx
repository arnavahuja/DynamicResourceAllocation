import { NavLink } from "react-router-dom";

const Icon = ({ d }) => (
  <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
);

const links = [
  { to: "/", label: "Train", d: "M3 3v18h18 M7 14l4-4 4 4 5-7" },
  { to: "/monitor", label: "Live Monitor", d: "M22 12h-4l-3 9L9 3l-3 9H2" },
  { to: "/results", label: "Results", d: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z" },
  { to: "/compare", label: "Compare", d: "M4 19V5 M10 19V9 M16 19V13 M22 19V7 M3 19h20" },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <h1>Cloud RL</h1>
      <p className="tagline">Scheduler · v1.0.0</p>
      <nav>
        {links.map((l) => (
          <NavLink key={l.to} to={l.to} end={l.to === "/"}>
            <Icon d={l.d} />
            {l.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
