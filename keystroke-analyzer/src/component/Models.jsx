import React, { useContext, useEffect } from 'react';
import { FileContext } from './FileContext';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import ImageSection from './ImageSection'; 


const Models = () => {
    const { analysisResult } = useContext(FileContext);
    const { rocCurve, comparison } = analysisResult;

    if (!analysisResult) {
        return <div>No report data available</div>;
    }

   
    return (
        <div>
            <div
                style={{
                    display: 'flex',
                    justifyContent: 'center',
                    //alignItems: 'center',
                }}
            >
                <div
                    style={{
                        display: 'flex',
                        flexDirection: 'row',
                        justifyContent: 'space-between',
                        marginTop: '10px',
                        width: '80%'
                    }}
                >
                    <ImageSection title="Micro-ROC Curve for All Models" image={rocCurve} />  
                    <ImageSection title="Accuracy of All Models" image={comparison} />
                </div>
            </div>
        </div>
    );
};

export default Models;
