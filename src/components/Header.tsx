import React from "react";
import "./Header.css";

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="header-content">
        <h1 className="header-title">
          <span className="title-icon">⚡</span>
          Stock Predictor
        </h1>
        <nav className="nav-links">
          <a href="#home" className="nav-link active">
            Home
          </a>
        </nav>
      </div>
    </header>
  );
};

export default Header;
