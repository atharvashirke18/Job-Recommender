import './App.css';
import Navbar from './Navbar';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages'
import JobRecommendation from './pages/JobRecommendation';
import SkillAnalysis from './pages/SkillAnalysis';
import MyForm from './components/Form';

function App() {
  return (
      <Router>
        <Navbar />
        <Routes>
          <Route path='/' element={<Home />} />
          {/* <Route path='/JobRecommendation' element={<JobRecommendation/>} /> */}
          {/* <Route path='/SkillAnalysis' element={<SkillAnalysis/>} /> */}

        </Routes>
      </Router>
  );
}

export default App;
