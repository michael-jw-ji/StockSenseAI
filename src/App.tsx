import React, { useState } from "react";
import "./App.css";
import StockPredictor from "./components/StockPredictor";
import Header from "./components/Header";

function App() {
  return (
    <div className="App">
      <Header />
      <main className="main-content">
        <StockPredictor />
      </main>
    </div>
  );
}

export default App;
