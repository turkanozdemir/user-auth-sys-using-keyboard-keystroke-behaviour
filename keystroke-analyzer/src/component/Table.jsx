import React, { useState, useEffect } from 'react';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import Papa from 'papaparse';

const Table = ({ file }) => {
  const [rowData, setRowData] = useState([]);
  const [paginationPageSize, setPaginationPageSize] = useState(20);

  useEffect(() => {
    if (file) {
      parseCsv(file);
    }
  }, [file]);

  const parseCsv = async (csvFile) => {
    const config = {
      complete: (result) => {
        const data = result.data.slice(1);
        setRowData(data);
      },
      header: true,
    };
    Papa.parse(csvFile, config);
  };

  const columnDefs = Object.keys(rowData[0] || {}).map((key) => ({
    headerName: key,
    field: key,
  }));

  const onGridReady = (params) => {
    params.api.paginationSetPageSize(paginationPageSize);
  };

  return (
    <div className="ag-theme-alpine" style={{ height: 400, width: '100%' }}>
      <AgGridReact
        columnDefs={columnDefs}
        rowData={rowData}
        domLayout='autoHeight'
        pagination={true}
        paginationPageSize={paginationPageSize}
        onGridReady={onGridReady}
      />
    </div>
  );
};

export default Table;
