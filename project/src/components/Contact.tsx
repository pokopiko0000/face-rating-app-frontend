import React from 'react';
import { useContactForm } from '../hooks/useContactForm';
import ContactHeader from './ContactHeader';
import ContactForm from './ContactForm';
import ContactStatusMessage from './ContactStatusMessage';
import ContactFAQ from './ContactFAQ';

export default function Contact() {
  const {
    formData,
    isSubmitting,
    submitStatus,
    handleInputChange,
    handleSubmit,
    resetForm
  } = useContactForm();

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <ContactHeader />
        
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 md:p-12">
          <ContactStatusMessage 
            status={submitStatus} 
            onReset={resetForm} 
          />
          
          {submitStatus === 'idle' && (
            <ContactForm
              formData={formData}
              isSubmitting={isSubmitting}
              onInputChange={handleInputChange}
              onSubmit={handleSubmit}
            />
          )}
        </div>

        <ContactFAQ />
      </div>
    </div>
  );
} 