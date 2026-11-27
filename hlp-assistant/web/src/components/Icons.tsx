import type { ReactNode, SVGProps } from 'react';

const base = (d: ReactNode) => (p: SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" width={20} height={20} fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden {...p}>
    {typeof d === 'string' ? <path d={d} /> : d}
  </svg>
);

export const IconHome = base('M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z');
export const IconChart = base('M4 20V10M10 20V4M16 20v-7M22 20H2');
export const IconUser = base(<><circle cx="12" cy="8" r="4" /><path d="M4 21c0-4 4-6 8-6s8 2 8 6" /></>);
export const IconSend = base('M4 12l16-8-6 16-2-7z');
export const IconThumbUp = base('M7 10v10H4V10zM7 10l4-7c1.5 0 2.5 1 2.2 2.6L12.6 9H19a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 17.8 20H7');
export const IconThumbDown = base('M7 14V4H4v10zM7 14l4 7c1.5 0 2.5-1 2.2-2.6L12.6 15H19a2 2 0 0 0 2-2.3l-1.2-7A2 2 0 0 0 17.8 4H7');
export const IconCopy = base(<><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a1 1 0 0 1 1-1h9" /></>);
export const IconGlobe = base(<><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18" /></>);
export const IconBook = base('M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM4 5v16M8 7h7');
export const IconPlus = base('M12 5v14M5 12h14');
export const IconTrash = base('M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13');
export const IconLogout = base('M15 4h4v16h-4M10 8l-4 4 4 4M6 12h10');
export const IconCheck = base('M5 12l5 5L20 7');
export const IconShield = base('M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z');
export const IconSpark = base('M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6');
export const IconEdit = base('M4 20h4L19 9l-4-4L4 16zM14 6l4 4');
export const IconX = base('M6 6l12 12M18 6L6 18');
