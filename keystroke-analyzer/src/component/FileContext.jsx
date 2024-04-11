// FileContext.js
import React, { createContext, useState } from 'react';

const FileContext = createContext();

const FileProvider = ({ children }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState({
    correlationMatrixImage: null,
    featureImportanceImage: null,
  });
  const [selectedModel, setSelectedModel] = useState('KNN'); // Set the default model here

  const setFile = (file) => {
    setSelectedFile(file);
  };

  const setResult = (result) => {
    setAnalysisResult(result);
  };

  const setModel = (model) => {
    setSelectedModel(model);
  };

  return (
    <FileContext.Provider value={{ selectedFile, setFile, analysisResult, setResult, selectedModel, setModel }}>
      {children}
    </FileContext.Provider>
  );
};

export { FileContext, FileProvider };
