import { cn } from '@/lib/utils';
import { Stethoscope } from 'lucide-react';

export function Logo({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center justify-center p-2 rounded-full bg-primary/20 text-primary", className)}>
        <Stethoscope size={32}/>
    </div>
  );
}
