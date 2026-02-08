import React from 'react';
//import './Navbar.css';
import {Nav , NavLink, NavMenu } from './components/Navbar/NavbarElements';

const Navbar = () => {
  return (
    <Nav>
      <NavMenu>
        <NavLink to="/">Home</NavLink>
        <NavLink to="/JobRecommendation">Job Recommendation</NavLink>
        <NavLink to="/SkillAnalysis">Skill Analysis</NavLink>
      </NavMenu>
    </Nav>
  );
};

export default Navbar;