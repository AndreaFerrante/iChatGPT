import './App.css'
import { makeStyles } from '@mui/styles';
import React, { useState } from 'react';
import { Box, Container } from '@mui/material';
import ChatWindow from './components/ChatWindow';
import MessageInput from './components/MessageInput';
import UploadButton from './components/UploadButton';
import LoadingIndicator from './components/LoadingIndicator';
import UploadSuccessModal from './components/UploadSuccessModal';

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
      }, 2000);
    }
  };

  const handleCloseModal = () => {
    setModalOpen(false);
  };

  return (
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
        maxWidth="sm"
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
  );
};

export default App;
