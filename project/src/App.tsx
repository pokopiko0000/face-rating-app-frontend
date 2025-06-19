import React from 'react';
import { Routes, Route } from 'react-router-dom';
import HomePage from './components/HomePage';
import PrivacyPolicy from './components/PrivacyPolicy';
import Contact from './components/Contact';
import TermsOfService from './components/TermsOfService';

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/privacy" element={<PrivacyPolicy />} />
      <Route path="/contact" element={<Contact />} />
      <Route path="/terms" element={<TermsOfService />} />
    </Routes>
  );
}

export default App;