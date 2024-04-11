import React, { useState } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import './assets/Navbar.css';

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const isHomePage = location.pathname === '/';

  const toggleDropdown = () => {
    setDropdownOpen(!dropdownOpen);
  };

  if (isHomePage) {
    return null;
  }

  return (
    <nav className="navbar">
      <ul className="navbar-list">
        <li className="navbar-logo">
          <Link to="/" className="logo-link">
            KEYSTROKE ANALYZER
          </Link>
        </li>
        <li className="navbar-item">
          <button className="navbar-button" onClick={toggleDropdown}>
            <span >About Dataset</span>
          </button>
          {dropdownOpen && (
            <ul className="dropdown">
              <li>
                <button
                  className="navbar-button"
                  onClick={() => navigate('/dataset')}
                >
                  <span className="larger-text">Dataset</span>
                </button>
              </li>
              <li>
                <button
                  className="navbar-button"
                  onClick={() => navigate('/features')}
                >
                  <span className="larger-text">Features</span>
                </button>
              </li>
            </ul>
          )}
        </li>
        <li className="navbar-item">
          <button
            className="navbar-button"
            onClick={() => navigate('/report')}
          >
            <span className="larger-text">Analyzing Result</span>
          </button>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
