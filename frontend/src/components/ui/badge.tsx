import { cn } from "@/lib/utils";

export function Badge({
  className,
  variant = "default",
  ...props
}: React.HTMLAttributes<HTMLDivElement> & {
  variant?: "default" | "secondary" | "outline" | "success" | "warning" | "danger";
}) {
  const variants = {
    default: "bg-primary/20 text-primary border-primary/30",
    secondary: "bg-secondary text-secondary-foreground border-border",
    outline: "border-border text-foreground",
    success: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    warning: "bg-amber-500/20 text-amber-300 border-amber-500/30",
    danger: "bg-red-500/20 text-red-300 border-red-500/30",
  };
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium",
        variants[variant],
        className
      )}
      {...props}
    />
  );
}
