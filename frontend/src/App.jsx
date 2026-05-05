import { Route, Routes, useLocation } from "react-router-dom";

import Sidebar from "./components/Layout/Sidebar.jsx";
import TopBar from "./components/Layout/TopBar.jsx";
import TrainPage from "./pages/TrainPage.jsx";
import CMDPTrainPage from "./pages/CMDPTrainPage.jsx";
import CMDPResultsPage from "./pages/CMDPResultsPage.jsx";
import MonitorPage from "./pages/MonitorPage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";
import ComparePage from "./pages/ComparePage.jsx";
import SweepPage from "./pages/SweepPage.jsx";
import SweepResultsPage from "./pages/SweepResultsPage.jsx";

const TITLES = {
  "/": "Train",
  "/cmdp": "CMDP Training",
  "/cmdp/results": "CMDP Results",
  "/sweep": "Parameter Sweep",
  "/sweep/results": "Sweep Results",
  "/monitor": "Live Monitor",
  "/results": "Results",
  "/compare": "Compare Runs",
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
            <Route path="/cmdp" element={<CMDPTrainPage />} />
            <Route path="/cmdp/results" element={<CMDPResultsPage />} />
            <Route path="/sweep" element={<SweepPage />} />
            <Route path="/sweep/results" element={<SweepResultsPage />} />
            <Route path="/monitor" element={<MonitorPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/compare" element={<ComparePage />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}
