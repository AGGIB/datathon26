import type { ReactNode } from 'react';
import './splash-screen.css';

export function SplashScreen({ children }: { children: ReactNode }) {
  return (
    <>
      <div className="splash-content">{children}</div>
      <div className="splash-screen" aria-hidden="true">
        <div className="splash-pill">
          <div className="splash-pill__fill" />
          <div className="splash-pill__mark splash-pill__mark--forte">
            <img src="/forte-logo.svg" alt="" />
          </div>
          <div className="splash-pill__mark splash-pill__mark--light">
            <img src="/forte-logo.svg" alt="" />
          </div>
        </div>
      </div>
    </>
  );
}
