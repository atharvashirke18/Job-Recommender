import logo from './logo.svg';
import './App.css';
import Navbar from './Navbar';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
// import Home from './Home'
import JobRecommendation from './components/JobRecommendation';
// import SkillAnalysis from './components/SkillAnalysis';

function App() {
  return (
      <Router>
        <Navbar />
        <Routes>
          <Route path='/components/JobRecommendation' element={<NavLink/>} />

        </Routes>
      </Router>
  );
}

export default App;
