"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Image from 'next/image';

const API_BASE_URL = 'http://localhost:8000';

export default function Home() {
  const [frame, setFrame] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastAction, setLastAction] = useState<string>('N/A');
  const isLoadingFrame = useRef<boolean>(false);

  const fetchInitialFrame = useCallback(async () => {
    setError(null);
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/start`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setFrame(`data:image/png;base64,${data.frame}`);
    } catch (e: unknown) {
      if (e instanceof Error) {
        setError(`Failed to fetch initial frame: ${e.message}`);
      } else {
        setError('An unknown error occurred during initial frame fetch.');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  const predictNextFrame = useCallback(async (action: string) => {
    if (!frame || isLoadingFrame.current) {
      return;
    }
    
    isLoadingFrame.current = true;
    setLastAction(action);
    setError(null);

    try {
      // remove 'data:image/png;base64,' prefix
      const base64Frame = frame.split(',')[1];
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ current_frame: base64Frame, action }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP error! status: ${response.status} - ${errorData.detail}`);
      }

      const data = await response.json();
      setFrame(`data:image/png;base64,${data.next_frame}`);
    } catch (e: unknown) {
      if (e instanceof Error) {
        setError(`Failed to predict next frame: ${e.message}`);
      } else {
        setError('An unknown error occurred during frame prediction.');
      }
    } finally {
      isLoadingFrame.current = false;
    }
  }, [frame]);

  useEffect(() => {
    fetchInitialFrame();
  }, [fetchInitialFrame]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const validKeys = ['w', 'a', 's', 'd', 'ArrowLeft', 'ArrowRight', ' '];
      if (validKeys.includes(event.key)) {
        event.preventDefault();
        predictNextFrame(event.key);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [predictNextFrame]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-900 text-white p-8">
      <h1 className="text-4xl font-bold mb-4">Tiny Engine</h1>
      <p className="text-lg mb-6">Play the game by using your keyboard.</p>
      
      <div className="w-full max-w-2xl border-4 border-gray-600 rounded-lg overflow-hidden shadow-2xl bg-black">
        <div className="aspect-w-4 aspect-h-3">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p>Loading game...</p>
            </div>
          ) : frame ? (
            <Image src={frame} alt="Game frame" fill className="object-contain" />
          ) : (
            <div className="flex items-center justify-center h-full">
              <p>Could not load game. Please try again.</p>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-800 border border-red-600 rounded-md text-white">
          <p><strong>Error:</strong> {error}</p>
          <button 
            onClick={fetchInitialFrame}
            className="mt-2 px-4 py-2 bg-red-600 hover:bg-red-500 rounded font-bold"
          >
            Restart Game
          </button>
        </div>
      )}

      <div className="mt-8 text-center">
        <h2 className="text-2xl font-semibold mb-2">Controls</h2>
        <div className="grid grid-cols-3 gap-2 justify-center font-mono text-lg">
            <div className="col-start-2 p-3 bg-gray-700 rounded-md">W</div>
            <div className="p-3 bg-gray-700 rounded-md">A</div>
            <div className="p-3 bg-gray-700 rounded-md">S</div>
            <div className="p-3 bg-gray-700 rounded-md">D</div>
            <div className="col-span-3 p-3 bg-gray-700 rounded-md">[SPACE] - Attack</div>
            <div className="col-span-3 p-3 bg-gray-700 rounded-md">Arrow Keys - Turn</div>
        </div>
         <p className="mt-4 text-gray-400">Last action: <span className="font-bold text-green-400">{lastAction}</span></p>
      </div>

      <div className="mt-8 text-center text-gray-400 text-md">
        <p>
          Follow the project on{' '}
          <a href="https://github.com/RevanthGundala/tiny_engine" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
            GitHub
          </a>{' '}
          and checkout the blog post on{' '}
          <a href="https://rgundal2.substack.com/p/diffusion-models-are-game-engines" target="_blank" rel="noopener noreferrer" className="text-orange-400 hover:underline">
            Substack
          </a>
          .
        </p>
      </div>
    </main>
  );
}
