import { Card } from "@/components/ui/card";

export function InfoCard() {
  return (
    <Card className="p-8 shadow-2xl border-0 bg-gradient-to-br from-orange-50 via-amber-50 to-orange-50 backdrop-blur-sm">
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <div className="h-1 rounded-full"></div>
          <h3 className="text-xl font-bold text-gray-900">
            How It Works
          </h3>
        </div>
        <div className="grid gap-4 mt-6">
          <StepCard
            number={1}
            title="Upload or Capture"
            description="Select an image from your device or take a photo with your camera"
          />
          <StepCard
            number={2}
            title="AI Classification"
            description="Our deep learning model analyzes the image for fire, smoke, or safe conditions"
          />
          <StepCard
            number={3}
            title="Threat Localization"
            description="If a threat is detected, bounding boxes highlight fire and smoke areas in the image"
          />
          <StepCard
            number={4}
            title="Rapid Response"
            description="Results enable faster emergency response thus minimizing environmental damage"
          />
        </div>
      </div>
    </Card>
  );
}

interface StepCardProps {
  number: number;
  title: string;
  description: string;
}

function StepCard({ number, title, description }: StepCardProps) {
  return (
    <div className="flex items-start gap-4 p-4 bg-white/70 rounded-xl shadow-sm border border-orange-200/50 hover:shadow-md transition-all">
      <div className="bg-orange-400 text-white rounded-full w-9 h-9 flex items-center justify-center shrink-0 shadow-sm font-semibold">
        {number}
      </div>
      <div>
        <p className="font-semibold text-gray-900 mb-1">{title}</p>
        <p className="text-sm text-gray-700">{description}</p>
      </div>
    </div>
  );
}