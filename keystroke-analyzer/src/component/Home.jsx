import React, { useState, useContext } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom'
import { FileContext } from './FileContext';
import WarningDialog from './WarningDialog'
import {
  Button,
  MenuItem,
  Paper,
  Typography,
  FormControl,
  createTheme,
  ThemeProvider,
  TextField,
} from '@mui/material';

axios.defaults.xsrfHeaderName = 'X-CSRFTOKEN';
axios.defaults.xsrfCookieName = 'csrftoken';

const Home = () => {
  const [file, setFile] = useState();
  const [fileName, setFileName] = useState('');
  const [splitRatio, setSplitRatio] = useState(0.2);
  const [selectedModel, setSelectedModel] = useState('MLP');
  const fileInputRef = React.createRef();
  const { setFile: setContextFile, setResult, setModel } = useContext(FileContext);
  const navigate = useNavigate();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [warningMessage, setWarningMessage] = useState('');


  const checkFileExtension = (file) => {
    const allowedExtensions = /(\.csv)$/i;
    return allowedExtensions.exec(file.name);
  };

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];

    if (uploadedFile) {
      if (!checkFileExtension(uploadedFile)) {
        setWarningMessage("Please select a file has CSV extension!");
        setIsDialogOpen(true);
        return;
      }

      setFile(uploadedFile);
      setFileName(uploadedFile.name);
      setContextFile(uploadedFile);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];

    if (droppedFile) {
      if (!checkFileExtension(droppedFile)) {
        setWarningMessage("Please select a file has CSV extension.");
        setIsDialogOpen(true);
        return;
      }

      setFile(droppedFile);
      setFileName(droppedFile.name);
      setContextFile(droppedFile);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleFileUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleAnalysis = async () => {
    if (!file) {
      setWarningMessage("Please load a CSV file!");
      setIsDialogOpen(true);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('splitRatio', splitRatio);

    if (selectedModel === 'SVM') {
      await svmModel(formData);
    }
    else if (selectedModel === 'MLP') {
      await mlpModel(formData);
    }
    else if (selectedModel === 'KNN') {
      await knnModel(formData)
    }
    else if (selectedModel === 'LR') {
      await lrModel(formData)
    }
    else if (selectedModel === 'All') {
      await models(formData)
    }

    setModel(selectedModel)
  };

  const svmModel = async (formData) => {
    await axios.post('http://localhost:8000/svm-predictions/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then((response) => {
        console.log('File uploaded successfully:', response);
        setResult(response.data);
        navigate('/report');
      })
      .catch((error) => {
        console.error('Error uploading file:', error);
      });
  };

  const mlpModel = async (formData) => {
    await axios.post('http://localhost:8000/mlp-predictions/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then((response) => {
        console.log('File uploaded successfully:', response);
        setResult(response.data);
        navigate('/report');
      })
      .catch((error) => {
        console.error('Error uploading file:', error);
      });
  };

  const knnModel = async (formData) => {
    await axios.post('http://localhost:8000/knn-predictions/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then((response) => {
        console.log('File uploaded successfully:', response);
        setResult(response.data);
        navigate('/report');
      })
      .catch((error) => {
        console.error('Error uploading file:', error);
      });
  };

  const lrModel = async (formData) => {
    await axios.post('http://localhost:8000/lr-predictions/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then((response) => {
        console.log('File uploaded successfully:', response);
        //navigate('/report', { state: { report: response.data } });
        setResult(response.data);
        navigate('/report');
      })
      .catch((error) => {
        console.error('Error uploading file:', error);
      });
  };

  const models = async (formData) => {
    await axios.post('http://localhost:8000/all-models/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then((response) => {
        console.log('File uploaded successfully:', response);
        setResult(response.data);
        navigate('/models');
      })
      .catch((error) => {
        console.error('Error uploading file:', error);
      });
  };


  const theme = createTheme({
    palette: {
      primary: {
        main: '#1976d2',
      },
      secondary: {
        main: '#f50057',
      },
    },
  });

  return (
    <ThemeProvider theme={theme}>
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundImage: `url(background.jpg)`,
        backgroundSize: 'cover',
      }}>

        <Paper style={{ padding: '20px', maxWidth: '400px', width: '100%', backgroundColor: 'rgba(255, 255, 255, 0.8)' }}>
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              border: '2px dashed #1976d2',
              padding: '10px',
              marginBottom: '10px',
              textAlign: 'center',
            }}
          >
            <span style={{ fontSize: '18px', marginBottom: '10px' }}>Drop a CSV file</span>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
            {fileName && <Typography variant="body1">{fileName}</Typography>}
          </div>
          <div
            style={{
              flexDirection: 'column',
              alignItems: 'center',
              padding: '10px',
              marginTop: '1px',
              marginBottom: '10px',
              textAlign: 'center',
            }}
          >
            <Button
              variant="contained"
              color="primary"
              onClick={handleFileUploadClick}
            >
              Upload Dataset
            </Button>
          </div>

          <FormControl fullWidth style={{ marginBottom: '10px' }}>
            <TextField
              select
              label="Model"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              fullWidth
              variant="outlined"
              margin="normal"
            >
              <MenuItem value="MLP">Multi-Layered Perceptron (MLP)</MenuItem>
              <MenuItem value="KNN">K-Nearest Neighbors (KNN)</MenuItem>
              <MenuItem value="SVM">Support Vector Machine (SVM)</MenuItem>
              <MenuItem value="LR">Logistic Regression (LR)</MenuItem>
              <MenuItem value="All">All Models</MenuItem>
            </TextField>
          </FormControl>

          <FormControl fullWidth style={{ marginBottom: '20px' }}>
            <TextField
              select
              label="Split Ratio"
              value={splitRatio}
              onChange={(e) => setSplitRatio(e.target.value)}
              fullWidth
              variant="outlined"
              margin="normal"
            >
              <MenuItem value={0.2}>20% Test - 80% Train</MenuItem>
              <MenuItem value={0.3}>30% Test - 70% Train</MenuItem>
              <MenuItem value={0.4}>40% Test - 60% Train</MenuItem>
              <MenuItem value={0.5}>50% Test - 50% Train</MenuItem>
            </TextField>
          </FormControl>
          <Button variant="contained" color="primary" fullWidth onClick={handleAnalysis}>
            Analyze
          </Button>
          <WarningDialog isOpen={isDialogOpen} onClose={() => setIsDialogOpen(false)} message={warningMessage} />
        </Paper>
      </div>
    </ThemeProvider>
  );
};

export default Home;
