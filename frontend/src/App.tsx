import React, { useState } from 'react'
import ChatInterface from './components/chat';
import FileUpload from './components/upload';
import { Box, Typography } from '@mui/material';


const App = () => {
    const [isUploaded, setIsUploaded] = useState(false);

    const handleUpload = () => {
        setIsUploaded(true);
    };

    return (
        <Box sx={{ padding: '20px' }}>
            <Typography variant="h4" sx={{ textAlign: 'center' }}>Chat with your PDF Documents</Typography>
            {!isUploaded && <FileUpload onUpload={handleUpload} />}
            {isUploaded && <ChatInterface />}
        </Box>
    );
};

export default App;
