import React, { useContext, useEffect } from 'react';
import { FileContext } from './FileContext';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import ImageSection from './ImageSection';


const AnalysisComponent = () => {
    const { selectedModel, analysisResult } = useContext(FileContext);
    const { confusionMatrix, rocCurve } = analysisResult;

    if (!analysisResult) {
        return <div>No report data available</div>;
    }

    const columnDefs = [
        { headerName: 'Metric', field: 'metric', width: '250%' },
        { headerName: 'Value', field: 'value', width: '255%' },
    ];

    const rowData = Object.keys(analysisResult.analyzeReport).map((metric) => ({
        metric,
        value: analysisResult.analyzeReport[metric],
    }));

    const renderTableTitle = () => {
        let title = '';
        if (selectedModel === 'SVM') {
            title = 'Support Vector Machine Model';
        } else if (selectedModel === 'MLP') {
            title = 'Multi-Layered Perceptron Model';
        } else if (selectedModel === 'KNN') {
            title = 'K-Nearest Neighbor Model';
        } else if (selectedModel === 'LP') {
            title = 'Logistic Regression Model';
        }
        return <h2 style={{ textAlign: 'left' }}>{title}</h2>;
    };

    return (
        <div>
            <div
                style={{
                    display: 'flex',
                    justifyContent: 'center',
                    minHeight: '100vh',
                    //alignItems: 'center',
                }}
            >
                <div style={{ width: '80%', }}>
                    <div style={{ marginBottom: '20px' }}>
                        {renderTableTitle()}
                    </div>

                    <div style={{
                        display: 'flex',
                    }}>
                        <ImageSection title="Confusion Matrix" image={confusionMatrix} />

                        <div className="ag-theme-alpine" style={{ height: 300, width: '45%' }}>
                            <AgGridReact
                                columnDefs={columnDefs}
                                rowData={rowData}
                                domLayout='autoHeight'
                            />
                            <ImageSection title="ROC Curve" image={rocCurve} />
                        </div>


                    </div>
                </div>
            </div>
        </div>
    );
};

export default AnalysisComponent;
