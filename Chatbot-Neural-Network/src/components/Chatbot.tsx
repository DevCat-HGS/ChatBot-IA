import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { prepareTrainingData, createAndTrainModel, predictResponse } from '../lib/model';

export default function Chatbot() {
  const [messages, setMessages] = useState<Array<{ text: string; isBot: boolean }>>([]);
  const [input, setInput] = useState('');
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [modelData, setModelData] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const initModel = async () => {
      const data = prepareTrainingData();
      const trainedModel = await createAndTrainModel(data.trainingData);
      setModel(trainedModel);
      setModelData(data);
      setMessages(prev => [...prev, { text: 'Hello! How can I help you?', isBot: true }]);
    };

    initModel();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !model || !modelData) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { text: userMessage, isBot: false }]);
    setInput('');

    const botResponse = predictResponse(userMessage, model, modelData.words, modelData.tags);
    setTimeout(() => {
      setMessages(prev => [...prev, { text: botResponse, isBot: true }]);
    }, 500);
  };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      <div className="flex-1 overflow-y-auto mb-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.isBot ? 'justify-start' : 'justify-end'}`}
          >
            <div
              className={`max-w-[70%] rounded-lg p-3 ${
                message.isBot
                  ? 'bg-gray-200 text-gray-800'
                  : 'bg-blue-500 text-white'
              }`}
            >
              {message.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Type your message..."
        />
        <button
          type="submit"
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          Send
        </button>
      </form>
    </div>
  );
}