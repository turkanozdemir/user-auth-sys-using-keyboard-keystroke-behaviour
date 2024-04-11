import React, { useContext, useEffect, useState } from 'react';
import { FileContext } from './FileContext';
import Table from './Table';

const DatasetComponent = () => {
    const { selectedFile } = useContext(FileContext);
    const [localSelectedFile, setLocalSelectedFile] = useState(null);

    useEffect(() => {
        // Set the localSelectedFile when the selectedFile changes
        setLocalSelectedFile(selectedFile);
    }, [selectedFile]);

    if (!localSelectedFile) {
        return <div>No data available</div>;
    }

    return (
        <div>
            <div
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                }}
            >
                <div style={{ width: '85%' }}>
                    <h2>Dataset</h2>
                    <Table file={localSelectedFile} />
                </div>
            </div>
        </div>
    );
};

export default DatasetComponent;
