"use client";

/**
 * The product's icon vocabulary: one set (Phosphor), one place.
 *
 * Every screen imports icons from here by what they MEAN in Aegis, not by what they
 * look like, so the set can be retuned without touching a view. Deep imports keep
 * each icon its own module: the bundle carries the ~60 used here, not all 1,500.
 *
 * Rules
 * - Default weight is "regular" (a 1.5 px stroke at 24 px): quiet beside Geist.
 * - The active sidebar item switches to "fill"; nothing else does.
 * - Sizes are 14 (inline with 12–13 px text), 16 (controls), 18 (navigation).
 * - Icons inherit currentColor; an icon is never the only carrier of a state.
 */
export type { Icon, IconProps, IconWeight } from "@phosphor-icons/react/dist/lib/types";

export { ArrowCounterClockwiseIcon as Reset } from "@phosphor-icons/react/dist/csr/ArrowCounterClockwise";
export { ArrowElbowDownLeftIcon as Enter } from "@phosphor-icons/react/dist/csr/ArrowElbowDownLeft";
export { ArrowLeftIcon as Back } from "@phosphor-icons/react/dist/csr/ArrowLeft";
export { ArrowRightIcon as Forward } from "@phosphor-icons/react/dist/csr/ArrowRight";
export { ArrowSquareOutIcon as External } from "@phosphor-icons/react/dist/csr/ArrowSquareOut";
export { ArrowUpRightIcon as OpenLink } from "@phosphor-icons/react/dist/csr/ArrowUpRight";
export { BedIcon as Bed } from "@phosphor-icons/react/dist/csr/Bed";
export { BellIcon as Bell } from "@phosphor-icons/react/dist/csr/Bell";
export { BellRingingIcon as BellRinging } from "@phosphor-icons/react/dist/csr/BellRinging";
export { BellSimpleRingingIcon as Alert } from "@phosphor-icons/react/dist/csr/BellSimpleRinging";
export { BinocularsIcon as Sentinel } from "@phosphor-icons/react/dist/csr/Binoculars";
export { BookOpenTextIcon as Source } from "@phosphor-icons/react/dist/csr/BookOpenText";
export { CaretDownIcon as CaretDown } from "@phosphor-icons/react/dist/csr/CaretDown";
export { CaretLeftIcon as CaretLeft } from "@phosphor-icons/react/dist/csr/CaretLeft";
export { CaretRightIcon as CaretRight } from "@phosphor-icons/react/dist/csr/CaretRight";
export { CaretUpDownIcon as CaretUpDown } from "@phosphor-icons/react/dist/csr/CaretUpDown";
export { CellSignalFullIcon as Signal } from "@phosphor-icons/react/dist/csr/CellSignalFull";
export { CheckIcon as Check } from "@phosphor-icons/react/dist/csr/Check";
export { ChecksIcon as Checks } from "@phosphor-icons/react/dist/csr/Checks";
export { CircleNotchIcon as Spinner } from "@phosphor-icons/react/dist/csr/CircleNotch";
export { ClockIcon as Clock } from "@phosphor-icons/react/dist/csr/Clock";
export { CpuIcon as Cpu } from "@phosphor-icons/react/dist/csr/Cpu";
export { DeviceMobileIcon as Pager } from "@phosphor-icons/react/dist/csr/DeviceMobile";
export { FlaskIcon as Flask } from "@phosphor-icons/react/dist/csr/Flask";
export { HeartIcon as Heart } from "@phosphor-icons/react/dist/csr/Heart";
export { HeartbeatIcon as Heartbeat } from "@phosphor-icons/react/dist/csr/Heartbeat";
export { HospitalIcon as Hospital } from "@phosphor-icons/react/dist/csr/Hospital";
export { ListIcon as Menu } from "@phosphor-icons/react/dist/csr/List";
export { LockSimpleIcon as Lock } from "@phosphor-icons/react/dist/csr/LockSimple";
export { MagnifyingGlassIcon as Search } from "@phosphor-icons/react/dist/csr/MagnifyingGlass";
export { MicrophoneIcon as Microphone } from "@phosphor-icons/react/dist/csr/Microphone";
export { MonitorIcon as Monitor } from "@phosphor-icons/react/dist/csr/Monitor";
export { PaperPlaneTiltIcon as Send } from "@phosphor-icons/react/dist/csr/PaperPlaneTilt";
export { PillIcon as Pill } from "@phosphor-icons/react/dist/csr/Pill";
export { PlugsConnectedIcon as Plugs } from "@phosphor-icons/react/dist/csr/PlugsConnected";
export { PlusIcon as Plus } from "@phosphor-icons/react/dist/csr/Plus";
export { PulseIcon as Pulse } from "@phosphor-icons/react/dist/csr/Pulse";
export { ScrollIcon as Journal } from "@phosphor-icons/react/dist/csr/Scroll";
export { ShieldCheckIcon as ShieldCheck } from "@phosphor-icons/react/dist/csr/ShieldCheck";
export { SidebarSimpleIcon as Sidebar } from "@phosphor-icons/react/dist/csr/SidebarSimple";
export { SignOutIcon as SignOut } from "@phosphor-icons/react/dist/csr/SignOut";
export { SlidersHorizontalIcon as Sliders } from "@phosphor-icons/react/dist/csr/SlidersHorizontal";
export { SquaresFourIcon as Overview } from "@phosphor-icons/react/dist/csr/SquaresFour";
export { StackIcon as Layers } from "@phosphor-icons/react/dist/csr/Stack";
export { TerminalWindowIcon as Terminal } from "@phosphor-icons/react/dist/csr/TerminalWindow";
export { TrayIcon as Inbox } from "@phosphor-icons/react/dist/csr/Tray";
export { TrendUpIcon as TrendUp } from "@phosphor-icons/react/dist/csr/TrendUp";
export { UserIcon as User } from "@phosphor-icons/react/dist/csr/User";
export { UsersThreeIcon as Staff } from "@phosphor-icons/react/dist/csr/UsersThree";
export { WarningIcon as Warning } from "@phosphor-icons/react/dist/csr/Warning";
export { WebhooksLogoIcon as Webhook } from "@phosphor-icons/react/dist/csr/WebhooksLogo";
export { WifiSlashIcon as Offline } from "@phosphor-icons/react/dist/csr/WifiSlash";
export { WindIcon as Wind } from "@phosphor-icons/react/dist/csr/Wind";
export { XIcon as Close } from "@phosphor-icons/react/dist/csr/X";
