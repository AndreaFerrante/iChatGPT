import { Widget, addResponseMessage, addUserMessage } from 'react-chat-widget';
import 'react-chat-widget/lib/styles.css';

const ChatInterface = () => {

    const handleNewUserMessage = (message:any) => {
        // Mock API call - replace with actual API call to OpenAI via FastAPI backend
        setTimeout(() => {
            addResponseMessage("This is a response based on your uploaded document.");
        }, 1000);
    };

    return (
        <div className="ChatInterface">
            <Widget
                handleNewUserMessage={handleNewUserMessage}
                title="Document Chatbot"
                subtitle="Chat with your PDF"
            />
        </div>
    );
};

export default ChatInterface;
