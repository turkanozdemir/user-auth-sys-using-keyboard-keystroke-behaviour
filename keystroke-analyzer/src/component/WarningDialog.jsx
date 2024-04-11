import React, { useState } from 'react';
import { Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions, Button } from '@mui/material';

const WarningDialog = ({ isOpen, onClose, message }) => {
    return (
        <Dialog open={isOpen} onClose={onClose}>
            <DialogTitle>Warning</DialogTitle>
            <DialogContent>
                <DialogContentText>
                    {message}
                </DialogContentText>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} color="primary">
                    OK
                </Button>
            </DialogActions>
        </Dialog>
    )
};

export default WarningDialog;