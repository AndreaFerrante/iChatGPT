import React from 'react';
import { Modal, Box, Typography, Button } from '@mui/material';

interface UploadSuccessModalProps {
  open: boolean;
  handleClose: () => void;
}

const UploadSuccessModal: React.FC<UploadSuccessModalProps> = ({ open, handleClose }) => {

  return (

    <Modal
      open={open}
      onClose={handleClose}
      aria-labelledby="modal-title"
      aria-describedby="modal-description"
    >
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 300,
          bgcolor: 'background.paper',
          borderRadius: '10px',
          boxShadow: 24,
          p: 4,
          textAlign: 'center',
        }}
      >
        <Typography id="modal-title" variant="h6" component="h2" gutterBottom>
          Upload and Embedding Successful!
        </Typography>
        <Typography id="modal-description" variant="body1" gutterBottom>
          Your document has been uploaded and processed.
        </Typography>
        <Button variant="contained" color="primary" onClick={handleClose}>
          Close
        </Button>
      </Box>
    </Modal>

  );
  
};

export default UploadSuccessModal;
