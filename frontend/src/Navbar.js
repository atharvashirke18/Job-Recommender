import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';
import { NavLink } from './components/Navbar/NavbarElements';

const Navbar = () => {
  return (
    <nav className="navbar">
      <ul className="nav-links">
        <li><Link to="/">Home</Link></li>
        <NavLink to="/components/JobRecommendation/index.js">Job Recommendation</NavLink>
        <li><Link to="/SkillAnalysis">Skill Analysis</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;