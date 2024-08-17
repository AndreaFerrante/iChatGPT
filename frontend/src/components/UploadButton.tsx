import React from 'react';
import { Button } from '@mui/material';

interface UploadButtonProps {
  handleUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
}

const UploadButton: React.FC<UploadButtonProps> = ({ handleUpload }) => {
  return (
    <Button
      variant="contained"
      component="label"
      style={{ marginTop: '10px' }}
    >
      Upload Document
      <input type="file" hidden onChange={handleUpload} />
    </Button>
  );
};

export default UploadButton;
