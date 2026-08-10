import { Check } from "lucide-react";

interface WizardStepperProps {
  currentStep: number;
  steps: string[];
}

export default function WizardStepper({ currentStep, steps }: WizardStepperProps) {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between relative">
        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-full h-[2px] bg-slate-100 -z-10" />
        {steps.map((step, index) => {
          const stepNum = index + 1;
          const isActive = currentStep === stepNum;
          const isCompleted = currentStep > stepNum;
          
          return (
            <div key={step} className="flex flex-col items-center bg-white px-2">
              <div 
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300
                  ${isActive ? 'bg-teal-600 text-white shadow-lg shadow-teal-500/30 scale-110' : 
                    isCompleted ? 'bg-teal-500 text-white' : 
                    'bg-slate-100 text-slate-400 border-2 border-slate-200'}`}
              >
                {isCompleted ? <Check className="w-5 h-5" /> : stepNum}
              </div>
              <span 
                className={`mt-2 text-xs font-semibold ${isActive ? 'text-teal-700' : isCompleted ? 'text-teal-600' : 'text-slate-400'}`}
              >
                {step}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
