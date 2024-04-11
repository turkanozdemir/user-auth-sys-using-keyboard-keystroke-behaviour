import React, { useContext, useEffect } from 'react';
import { FileContext } from './FileContext';
import ImageSection from './ImageSection';
import Table from './Table';
import { AgGridReact } from 'ag-grid-react';

const FeaturesComponent = () => {
    const { analysisResult, setResult } = useContext(FileContext);
    const rowData = [
        { Features: 'H', Descriptions: 'Hold, the average time passed between when a key is pressed and released' },
        { Features: 'DD', Descriptions: 'Down-Down, the average time passed between a key is pressed and the next key is pressed after it' },
        { Features: 'UD', Descriptions: 'Up-Down, the average time passed between a key is released and the next key is pressed after it' },
        { Features: 'key_stroke_average', Descriptions: 'Average number of the keys pressed within 500ms (half a second)' },
        { Features: 'back_space_count', Descriptions: 'Average number of the backspace pressed within 500ms (half a second)' },
        { Features: 'used_caps', Descriptions: 'Indicates if the user used caps lock to capitalize the letters' },
        { Features: 'shift_left_favored', Descriptions: 'Indicates if the user used left shift button or right shift button more' },
        { Features: 'label', Descriptions: 'The user who wrote the text' },
    ];

    const columnDefs = [
        { headerName: 'Features', field: 'Features' },
        { headerName: 'Descriptions', field: 'Descriptions', width: 750 },
    ];

    useEffect(() => {
        const storedAnalysisResult = JSON.parse(localStorage.getItem('analysisResult'));
        if (storedAnalysisResult) {
            setResult(storedAnalysisResult);
        }
    }, [setResult]);

    if (!analysisResult) {
        return <div>No analysis result available</div>;
    }

    const { correlationMatrixImage, featureImportanceImage } = analysisResult;

    if (!correlationMatrixImage && !featureImportanceImage) {
        return <div>No image data available</div>;
    }

    return (
        <div>
            <div
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    minHeight: '100vh',
                    alignItems: 'center',
                }}
            >
                <div style={{ width: '80%' }}>
                    <h2>Features Information</h2>
                    <div className="ag-theme-alpine" style={{ height: 400, width: '65%', marginTop: '20px', }}>
                        <AgGridReact
                            columnDefs={columnDefs}
                            rowData={rowData}
                            domLayout='autoHeight'
                        />
                    </div>
                </div>
                <div
                    style={{
                        display: 'flex',
                        flexDirection: 'row',
                        justifyContent: 'space-between',
                        marginTop: '10px',
                        width: '80%'
                    }}
                >
                    <div style={{ maxWidth: '75%' }}>
                        <h3>{"Feature Importance"}</h3>
                            <img
                            src={`/${'fi-dt.png'}`}
                          />
                    </div>                    
                    <ImageSection title="Correlation Matrix" image={correlationMatrixImage} />
                </div>
            </div>
        </div>
    );
};

export default FeaturesComponent;
