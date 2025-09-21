import * as React from "react"
import { cn } from "@/lib/utils"

const Loader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("loader", className)}
    {...props}
  />
))
Loader.displayName = "Loader"

export { Loader }
