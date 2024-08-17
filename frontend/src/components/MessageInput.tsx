import React from 'react';
import { Box, TextField, Button } from '@mui/material';

interface MessageInputProps {
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  handleSend: () => void;
}

const MessageInput: React.FC<MessageInputProps> = ({ input, setInput, handleSend }) => {
  return (
    <Box display="flex" alignItems="center" marginTop={2}>
      <TextField
        variant="outlined"
        fullWidth
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type a message..."
      />
      <Button variant="contained" color="primary" onClick={handleSend} style={{ marginLeft: '10px' }}>
        Send
      </Button>
    </Box>
  );
};

export default MessageInput;
