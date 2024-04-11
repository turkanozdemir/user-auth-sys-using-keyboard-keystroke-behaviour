import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/';

class ModelService {

    getModel(endpoint, formData) {
        try {
            const response = axios.post(API_BASE_URL + endpoint, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            console.log('File uploaded successfully:', response);
        } catch (error) {
            console.error('Error uploading file:', error);
        }

        return response.data;
    }
}

export default new ModelService();