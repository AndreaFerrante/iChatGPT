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
      height="610px"
      width="820px"
      overflow="auto"
      border={1}
      padding={1}
      borderRadius={2}
      bgcolor="whitesmoke"
    >
      {messages.map((msg, index) => (
        <Box
          key={index}
          margin={0.5}
          padding={1}
          borderRadius={2}
          alignSelf={msg.sender === 'user' ? 'flex-end' : 'flex-start'}
          bgcolor={msg.sender === 'user' ? 'lightblue' : 'lightgreen'}
          sx={{
            maxWidth: '75%',            // Limit the width of the chat bubble
            wordWrap: 'break-word',     // Ensure long words break correctly
            overflowWrap: 'break-word', // Handle long words or URLs
          }}
        >
          {msg.message}
        </Box>
      ))}
    </Box>

  );
};

export default ChatWindow;
