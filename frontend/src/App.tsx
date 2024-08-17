import './App.css'
import React, { useState } from 'react';
import { Box, Container } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import ChatWindow from './components/ChatWindow';
import MessageInput from './components/MessageInput';
import UploadButton from './components/UploadButton';
import LoadingIndicator from './components/LoadingIndicator';
import UploadSuccessModal from './components/UploadSuccessModal';


const theme = createTheme({

  typography: {
    fontSize: 11, // Default font size in px, which can be adjusted
    body1: {
      fontSize: '0.8rem', // Specific body text size
    },
    button: {
      fontSize: '1rem', // Smaller button text size
    }
    
  },

});


interface ChatMessage {
  sender: 'user' | 'bot';
  message: string;
}


const App: React.FC = () => {

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>('');
  const [uploading, setUploading] = useState<boolean>(false);
  const [modalOpen, setModalOpen] = useState<boolean>(false);

  const handleSend = () => {
    if (input.trim() === '') return;

    const userMessage: ChatMessage = { sender: 'user', message: input };
    setMessages([...messages, userMessage]);

    setInput('');

    // Mocked backend response
    const botResponse: ChatMessage = {
      sender: 'bot',
      message: `You said: ${input}`,
    };

    setMessages((prevMessages) => [...prevMessages, botResponse]);
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setUploading(true);

      // Mock the backend call
      setTimeout(() => {
        setUploading(false);
        setModalOpen(true);
      }, 1500);
    }
  };

  const handleCloseModal = () => {
    setModalOpen(false);
  };

  return (

    <ThemeProvider theme={theme}>

    <Box 
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '90vh',
        width: '50vw',
        bgcolor: 'darkgray', // Extend dark gray to the entire viewport
        overflow: 'hidden',
      }}
    >
      <Container
        maxWidth="md"
        sx={{
          bgcolor: 'lightgray',
          padding: '20px',
          borderRadius: '10px',
          boxShadow: 3,
        }}
      >
        {uploading ? (
          <LoadingIndicator />
        ) : (
          <>
            <ChatWindow messages={messages} />
            <MessageInput input={input} setInput={setInput} handleSend={handleSend} />
            <UploadButton handleUpload={handleUpload} />
          </>
        )}

        <UploadSuccessModal open={modalOpen} handleClose={handleCloseModal} />

      </Container>
    </Box>

    </ThemeProvider>

  );
};

export default App;
