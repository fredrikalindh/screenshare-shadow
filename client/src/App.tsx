import './App.css';

import {
  RTVIClientAudio,
  RTVIClientVideo,
  useRTVIClient,
  useRTVIClientTransportState,
} from '@pipecat-ai/client-react';
import { useEffect, useState } from 'react';

import { ConnectButton } from './components/ConnectButton';
import { DebugDisplay } from './components/DebugDisplay';
import { RTVIProvider } from './providers/RTVIProvider';
import { StatusDisplay } from './components/StatusDisplay';

function ScreenShareVideo() {
  const transportState = useRTVIClientTransportState();
  const isConnected = transportState !== 'disconnected';
  const rtviClient = useRTVIClient();
  const screen_sharing = rtviClient?.isSharingScreen
  const [isSharing, setIsSharing] = useState(screen_sharing);

  useEffect(() => {
    setIsSharing(screen_sharing);
  }, [screen_sharing]);

  if (!isConnected) return null;

  if (!isSharing) {
    return (
      <button onClick={() => rtviClient?.enableScreenShare(true)}>
        Share Screen
      </button>
    );
  }

  return (
    <div className="bot-container">
      <div className="video-container">
        <RTVIClientVideo trackType='screenVideo' participant="local" fit="cover" />
      </div>
      <button onClick={() => rtviClient?.enableScreenShare(false)}>Stop Sharing</button>
    </div>
  );
}

function AppContent() {
  return (
    <div className="app">
      <div className="status-bar">
        <StatusDisplay />
        <ConnectButton />
      </div>

      <div className="main-content">
        <ScreenShareVideo />
      </div>

      <DebugDisplay />
      <RTVIClientAudio />
    </div>
  );
}

function App() {
  return (
    <RTVIProvider>
      <AppContent />
    </RTVIProvider>
  );
}

export default App;
