import { Route, Routes, useLocation } from "react-router-dom";

import Sidebar from "./components/Layout/Sidebar.jsx";
import TopBar from "./components/Layout/TopBar.jsx";
import TrainPage from "./pages/TrainPage.jsx";
import MonitorPage from "./pages/MonitorPage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";

const TITLES = {
  "/": "Train",
  "/monitor": "Live Monitor",
  "/results": "Results",
};

export default function App() {
  const loc = useLocation();
  const title = TITLES[loc.pathname] ?? "Cloud RL Scheduler";
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main">
        <TopBar title={title} />
        <div className="page">
          <Routes>
            <Route path="/" element={<TrainPage />} />
            <Route path="/monitor" element={<MonitorPage />} />
            <Route path="/results" element={<ResultsPage />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}
