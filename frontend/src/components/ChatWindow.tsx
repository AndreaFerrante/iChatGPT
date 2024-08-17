import React from 'react';
import { Box } from '@mui/material';

interface ChatMessage {
  sender: 'user' | 'bot';
  message: string;
}

interface ChatWindowProps {
  messages: ChatMessage[];
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages }) => {
  return (
    <Box
      display="flex"
      flexDirection="column"
      height="400px"
      overflow="auto"
      border={1}
      padding={2}
      borderRadius={2}
      bgcolor="white"
    >
      {messages.map((msg, index) => (
        <Box
          key={index}
          margin={1}
          padding={2}
          borderRadius={10}
          alignSelf={msg.sender === 'user' ? 'flex-end' : 'flex-start'}
          bgcolor={msg.sender === 'user' ? 'lightblue' : 'lightgreen'}
        >
          {msg.message}
        </Box>
      ))}
    </Box>
  );
};

export default ChatWindow;
