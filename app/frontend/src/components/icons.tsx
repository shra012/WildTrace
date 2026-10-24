import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function make(paths: ReactNode, displayName: string) {
  function Icon({ size = 18, className, ...rest }: IconProps) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.9}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        className={className}
        {...rest}
      >
        {paths}
      </svg>
    );
  }
  Icon.displayName = displayName;
  return Icon;
}

export const SendIcon = make(<><path d="M4 12 20 4l-4 16-4-7-8-1Z" /><path d="m12 13 8-9" /></>, "SendIcon");
export const PaperclipIcon = make(<path d="m20 11-8.5 8.5a5 5 0 0 1-7-7L13 4a3.3 3.3 0 0 1 4.7 4.7l-8.4 8.5a1.7 1.7 0 0 1-2.4-2.4L14.5 7" />, "PaperclipIcon");
export const XIcon = make(<><path d="M6 6l12 12" /><path d="M18 6 6 18" /></>, "XIcon");
export const SunIcon = make(<><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>, "SunIcon");
export const MoonIcon = make(<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z" />, "MoonIcon");
export const ChipIcon = make(<><rect x="6" y="6" width="12" height="12" rx="2" /><rect x="9.5" y="9.5" width="5" height="5" rx="1" /><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" /></>, "ChipIcon");
export const RobotIcon = make(<><path d="M5 20h10" /><path d="M8 20v-4l3-5" /><circle cx="11" cy="11" r="1.6" /><path d="m12.3 10.1 4.7-3.6" /><circle cx="17.5" cy="6" r="1.4" /><path d="m18.5 7 1.8 3.5" /><path d="M19.5 10.5 21 13" /></>, "RobotIcon");
export const SparklesIcon = make(<><path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8Z" /><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8Z" /></>, "SparklesIcon");
export const ChatIcon = make(<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z" />, "ChatIcon");
export const PlayIcon = make(<path d="M7 4.5v15l12.5-7.5Z" />, "PlayIcon");
export const RefreshIcon = make(<><path d="M20 11a8 8 0 0 0-14.3-4.3L4 9" /><path d="M4 4v5h5" /><path d="M4 13a8 8 0 0 0 14.3 4.3L20 15" /><path d="M20 20v-5h-5" /></>, "RefreshIcon");
export const CheckIcon = make(<path d="m5 12.5 4.5 4.5L19 7.5" />, "CheckIcon");
export const AlertIcon = make(<><path d="M10.3 3.9 2.4 17.5A2 2 0 0 0 4.1 20.5h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4.5M12 17h.01" /></>, "AlertIcon");
export const ClockIcon = make(<><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>, "ClockIcon");
export const DownloadIcon = make(<><path d="M12 4v11" /><path d="m7 10 5 5 5-5" /><path d="M5 20h14" /></>, "DownloadIcon");
export const ExternalIcon = make(<><path d="M14 4h6v6" /><path d="m20 4-9 9" /><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" /></>, "ExternalIcon");
export const GridIcon = make(<><rect x="4" y="4" width="7" height="7" rx="1.5" /><rect x="13" y="4" width="7" height="7" rx="1.5" /><rect x="4" y="13" width="7" height="7" rx="1.5" /><rect x="13" y="13" width="7" height="7" rx="1.5" /></>, "GridIcon");
export const LayersIcon = make(<><path d="m12 3 9 5-9 5-9-5Z" /><path d="m3 13 9 5 9-5" /></>, "LayersIcon");
export const VideoIcon = make(<><rect x="3" y="6" width="13" height="12" rx="2" /><path d="m16 10 5-3v10l-5-3" /></>, "VideoIcon");
export const PenIcon = make(<><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z" /><path d="m14.5 5.5 3 3" /></>, "PenIcon");
export const WifiOffIcon = make(<><path d="M3 3l18 18" /><path d="M8.5 16.5a5 5 0 0 1 7 0" /><path d="M5 12.5a10 10 0 0 1 5-2.6M14 10a10 10 0 0 1 5 2.5" /><path d="M2 8.8a15 15 0 0 1 4.2-2.6M10.7 5a15 15 0 0 1 11.3 3.8" /><path d="M12 20h.01" /></>, "WifiOffIcon");
export const ChevronDownIcon = make(<path d="m6 9 6 6 6-6" />, "ChevronDownIcon");
export const TrashIcon = make(<><path d="M4 7h16" /><path d="M10 11v6M14 11v6" /><path d="M6 7l1 13h10l1-13" /><path d="M9 7V4h6v3" /></>, "TrashIcon");
export const UserIcon = make(<><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></>, "UserIcon");
export const PawIcon = make(<><ellipse cx="7" cy="9" rx="1.8" ry="2.4" /><ellipse cx="11" cy="6" rx="1.8" ry="2.4" /><ellipse cx="15.5" cy="6.5" rx="1.8" ry="2.4" /><ellipse cx="18.5" cy="10.5" rx="1.6" ry="2.2" /><path d="M8 17c0-3 2-5.5 4.5-5.5S17 14 17 17c0 2-1.5 3-3 3-1 0-1.5-.5-2-.5s-1 .5-2 .5c-1.5 0-2-1-2-3Z" /></>, "PawIcon");
export const LibraryIcon = make(<><rect x="4" y="4" width="4" height="16" rx="1" /><rect x="10" y="4" width="4" height="16" rx="1" /><path d="m16 5 3.8 1 -3.6 14-3.8-1" /></>, "LibraryIcon");
export const ActivityIcon = make(<path d="M3 12h4l3-8 4 16 3-8h4" />, "ActivityIcon");
export const ImageIcon = make(<><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="9" cy="10" r="2" /><path d="m21 17-5-5-9 8" /></>, "ImageIcon");
export const CircleIcon = make(<circle cx="12" cy="12" r="7" />, "CircleIcon");
export const MinusIcon = make(<path d="M6 12h12" />, "MinusIcon");
export const MaximizeIcon = make(<><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></>, "MaximizeIcon");
export const RouteIcon = make(<><circle cx="6" cy="18" r="2.2" /><circle cx="18" cy="6" r="2.2" /><path d="M8 18h7a3.5 3.5 0 0 0 0-7H9a3.5 3.5 0 0 1 0-7h7" /></>, "RouteIcon");
export const GaugeIcon = make(<><path d="M12 14l4-4" /><path d="M3.5 18a9 9 0 1 1 17 0" /></>, "GaugeIcon");
export const TargetIcon = make(<><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r=".6" /></>, "TargetIcon");
