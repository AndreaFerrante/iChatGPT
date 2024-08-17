import React, { useState } from 'react'
import { FilePond, registerPlugin, FilePondFile  } from 'react-filepond';
import 'filepond/dist/filepond.min.css';
import { Box, LinearProgress, Typography } from '@mui/material';

const FileUpload = ({ onUpload }) => {
    const [files, setFiles] = useState([]);
    const [uploading, setUploading] = useState(false);

    // Wrapping setFiles to ensure type compatibility
    const handleUpdateFiles = (fileItems: FilePondFile[]) => {
      setFiles(fileItems);
    };

    const handleProcessFiles = () => {
        setUploading(true);
        // Mock API call - replace with actual API call to backend
        setTimeout(() => {
            setUploading(false);
            onUpload();  // Notify parent component
        }, 2000);
    };

    return (
        <Box sx={{ width: '100%', textAlign: 'center', marginTop: '20px' }}>
            <FilePond
                files={files}
                onupdatefiles={handleUpdateFiles}
                allowMultiple={false}
                maxFiles={10}
                server={null} // Replace with actual server endpoint
                name="file"
                labelIdle='Drag & Drop your PDF or <span class="filepond--label-action">Browse</span>'
            />
            {uploading && (
                <Box sx={{ width: '100%', marginTop: '20px' }}>
                    <LinearProgress />
                    <Typography variant="body2" color="text.secondary">Embedding document...</Typography>
                </Box>
            )}
            {!uploading && files.length > 0 && (
                <button onClick={handleProcessFiles}>Upload & Embed</button>
            )}
        </Box>
    );

};

export default FileUpload;
