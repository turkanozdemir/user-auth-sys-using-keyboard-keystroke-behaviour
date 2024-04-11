import React from 'react';
import ReactDOM from 'react-dom';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import AnalysisComponent from './AnalysisComponent';

// Tema oluşturma
const theme = createTheme();

// Ana bileşeni tema ile sarmalama
const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <AnalysisComponent />
    </ThemeProvider>
  );
};

ReactDOM.render(<App />, document.getElementById('root'));
