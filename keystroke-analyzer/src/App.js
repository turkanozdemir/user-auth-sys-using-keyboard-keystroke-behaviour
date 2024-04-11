import React from 'react';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import Home from './component/Home';
import AnalysisComponent from './component/AnalysisComponent';
import DatasetComponent from './component/DatasetComponent';
import FeaturesComponent from './component/FeaturesComponent'
import Models from './component/Models'
import Navbar from './component/Navbar';
import { FileProvider } from './component/FileContext';

function App() {
  return (
    <FileProvider>
      <BrowserRouter>
        <div>
          <Navbar />
          <div>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/report" element={<AnalysisComponent />} />
              <Route path="/dataset" element={<DatasetComponent />} />
              <Route path="/features" element={<FeaturesComponent />} />
              <Route path="/models" element={<Models />} />
            </Routes>
          </div>
        </div>
      </BrowserRouter>
    </FileProvider>

  );
}

export default App;
