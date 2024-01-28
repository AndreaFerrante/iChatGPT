In this folder we do add testing for API connectio with a Javascript frontend.
Please, follow the following instructions:

Prerequisites:

1. Python (3.x recommended)
2. Node.js (LTS version recommended)
3. pip (Python package installer)
4. npm or yarn (Node.js package manager)


--------------------------------------------------------------------------------------------------------------------
---->> Create a new NextJS app as follows:

1. Open another terminal, navigate to another directory for your frontend and run: npx create-next-app my-nextjs-app
2. Navigate into your project's directory: cd my-nextjs-app
3. We'll use Axios to make API requests to our Flask backend, install it: npm install axios
4. Open pages/index.js and modify it to look like below:

	import axios from 'axios';
	import { useEffect, useState } from 'react';

	export default function Home() {
	  const [message, setMessage] = useState('');

	  useEffect(() => {
	    const fetchData = async () => {
	      const result = await axios.get('http://localhost:5000/api/hello');
	      setMessage(result.data.message);
	    };
	    fetchData();
	  }, []);

	  return (
	    <div>
	      <h1>{message}</h1>
	    </div>
	  );
	}



---->> Run the application:

1. Make sure your Flask app is running on http://localhost:5000.
2. In another terminal, navigate to your Next.js app directory and run:

	npm run dev

3. Now, if you navigate to http://localhost:3000, you should see a message fetched from your Flask API.
--------------------------------------------------------------------------------------------------------------------