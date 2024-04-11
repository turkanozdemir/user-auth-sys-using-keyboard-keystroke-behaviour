import React from 'react';

const ImageSection = ({ title, image }) => {
    return (
        <div style={{ maxWidth: '75%' }}>
            <h3>{title}</h3>
            {image && (
                <img
                    src={`data:image/png;base64,${image}`}
                    alt={title}
                    style={{ maxWidth: '100%', height: '100%' }}
                />
            )}
        </div>
    );
};

export default ImageSection;
