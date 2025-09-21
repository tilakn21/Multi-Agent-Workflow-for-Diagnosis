"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  BotMessageSquare,
  Brain,
  Stethoscope,
  Activity,
} from "lucide-react";
import { Logo } from "@/components/logo";
import { useState, useEffect, useRef } from "react";
import {
  motion,
  useScroll,
  useTransform,
  AnimatePresence,
} from "framer-motion";
import { useInView } from "framer-motion";
import { useToast } from "@/hooks/use-toast";
import { useChatHistory } from "@/hooks/use-chat-history";
import type { Message } from "@/lib/types";
import dynamic from "next/dynamic";

const DynamicBrainPointCloudModel = dynamic(
  () =>
    import("./brain-model-3d").then((mod) => mod.DynamicBrainPointCloudModel),
  { ssr: false }
);

export default function Home() {
  const [selectedPart, setSelectedPart] = useState("");
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isLoaded, setIsLoaded] = useState(false);

  const { scrollYProgress } = useScroll();
  const y = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 1.2]);

  useEffect(() => {
    setIsLoaded(true);

    const updateMousePosition = (e: any) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 2 - 1,
        y: -(e.clientY / window.innerHeight) * 2 + 1,
      });
    };

    window.addEventListener("mousemove", updateMousePosition);
    return () => window.removeEventListener("mousemove", updateMousePosition);
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.6,
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6 },
    },
  };

  // Define types for the parameters and hooks
  interface PatientInfo {
    temperature: number;
    heart_rate: number;
    blood_pressure: string;
    oxygen_saturation: number;
  }

  interface Payload {
    conversation: string;
    patient_info: PatientInfo;
    doctor_opinion: string;
    abha_id: string;
    image: null;
  }

  // Updated formatInputWithGroq to use API route
  const formatInputWithGroq = async (input: string): Promise<string> => {
    try {
      const response = await fetch("/api/groq-format", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
      });
      const data = await response.json();
      if (!data.content) throw new Error("Groq API returned null content");
      return data.content;
    } catch (error) {
      console.error("Error calling Groq API:", error);
      throw error;
    }
  };

  // Updated sendMessage to accept conversationId
  const sendMessage = async (
    message: string,
    transcriptionResult: string | null,
    conversationId: string
  ): Promise<void> => {
    const { toast } = useToast();
    const { addMessage } = useChatHistory();

    // Add a loading message
    const loadingId = Date.now().toString() + "-loading";
    addMessage(conversationId, {
      id: loadingId,
      sender: "bot",
      text: "Loading...",
    });

    try {
      // Use Groq to format the input before sending to ngrok API
      const formattedInput = await formatInputWithGroq(message);
      const payload = { symptomDescription: formattedInput };

      const response = await fetch(
        "https://5796209a4fff.ngrok-free.app/compose_case",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch API response");
      }

      const data = await response.json();

      // Add the actual response as a new message
      addMessage(conversationId, {
        id: Date.now().toString(),
        sender: "bot",
        text: data.response || "No response received.",
      });
    } catch (error) {
      console.error("Error in sendMessage:", error);
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive",
      });
      // Optionally add error message
      addMessage(conversationId, {
        id: Date.now().toString(),
        sender: "bot",
        text: "Error: Could not get response.",
      });
    }
  };

  // Updated ChatInterface component
  const ChatInterface = () => {
    const { messages, addMessage } = useChatHistory();
    const [input, setInput] = useState<string>("");
    const conversationId = "default-conversation"; // Example conversation ID

    const handleSend = (): void => {
      if (!input.trim()) return;

      addMessage(conversationId, {
        id: Date.now().toString(),
        sender: "user",
        text: input,
      });
      sendMessage(input, null, conversationId);
      setInput("");
    };

    return (
      <div className="chat-interface">
        <div className="chat-messages">
          {messages(conversationId).map((msg, index) => (
            <div
              key={msg.id || index}
              className={`message ${msg.sender === "user" ? "user" : "bot"}`}
            >
              {msg.text}
            </div>
          ))}
        </div>

        <div className="chat-input">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
          />
          <button onClick={handleSend}>Send</button>
        </div>
      </div>
    );
  };

  return (
    <>
      {/* Preloader */}
      <AnimatePresence>
        {!isLoaded && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-background"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="w-16 h-16 border-4 border-primary/20 border-t-primary rounded-full"
            />
          </motion.div>
        )}
      </AnimatePresence>

      <main className="relative min-h-screen w-full bg-background text-foreground overflow-x-hidden">
        {/* Animated Background Grid */}
        <div className="fixed inset-0 opacity-10">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#22c55e0a_1px,transparent_1px),linear-gradient(to_bottom,#22c55e0a_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />
        </div>

        {/* Floating Particles */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-green-400/30 rounded-full shadow-lg"
              animate={{
                y: [0, -20, 0],
                x: [0, Math.random() * 10 - 5, 0],
                opacity: [0.3, 1, 0.3],
              }}
              transition={{
                duration: Math.random() * 3 + 2,
                repeat: Infinity,
                delay: Math.random() * 2,
              }}
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
            />
          ))}
        </div>

        {/* Enhanced Header */}
        <motion.header
          initial={{ y: -100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="fixed top-0 z-40 flex w-full items-center justify-between p-4 md:p-6 backdrop-blur-lg bg-background/80 border-b border-green-400/20 shadow-md"
        >
          <motion.div
            className="flex items-center gap-3"
            whileHover={{ scale: 1.05 }}
          >
            <Logo className="h-10 w-10 text-green-400" />
            <h1 className="text-xl font-bold bg-gradient-to-r from-green-400 to-green-600 bg-clip-text text-transparent">
              MediMind
            </h1>
          </motion.div>

          <nav className="hidden items-center gap-6 md:flex">
            {["About", "Features", "Research", "Contact"].map((item) => (
              <motion.div key={item} whileHover={{ y: -2 }}>
                <Button
                  variant="ghost"
                  asChild
                  className="text-sm font-medium text-green-700"
                >
                  <Link href="#">{item}</Link>
                </Button>
              </motion.div>
            ))}
          </nav>

          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              asChild
              className="bg-gradient-to-r from-green-400 to-green-600 hover:from-green-500 hover:to-green-700 text-white"
            >
              <Link href="/chat">
                Launch Platform <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </motion.div>
        </motion.header>

        {/* Hero Section */}
        <section className="relative min-h-screen flex items-center justify-center pt-20">
          <motion.div
            className="container mx-auto px-4 grid lg:grid-cols-2 gap-12 items-center"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            {/* Content */}
            <motion.div
              className="relative z-20 space-y-8"
              style={{ y, opacity }}
            >
              <motion.div variants={itemVariants} className="space-y-6">
                <motion.h2
                  className="text-5xl sm:text-6xl lg:text-7xl font-bold leading-tight text-green-900"
                  style={{ scale }}
                >
                  <span className="block">Smarter</span>
                  <span className="block bg-gradient-to-r from-green-400 via-green-500 to-green-700 bg-clip-text text-transparent">
                    Medical AI,
                  </span>
                  <span className="block text-4xl sm:text-5xl lg:text-6xl">
                    Better Healthcare
                  </span>
                </motion.h2>

                <motion.p
                  className="text-lg lg:text-xl text-green-700 max-w-lg leading-relaxed"
                  variants={itemVariants}
                >
                  Revolutionizing healthcare with advanced AI diagnostics,
                  personalized treatment recommendations, and predictive
                  analytics for a healthier tomorrow.
                </motion.p>
              </motion.div>

              {/* CTA Buttons */}
              <motion.div
                className="flex flex-col sm:flex-row gap-4"
                variants={itemVariants}
              >
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    size="lg"
                    asChild
                    className="bg-gradient-to-r from-green-400 to-green-600 hover:from-green-500 hover:to-green-700 text-lg px-8 py-6 rounded-xl shadow-md text-white"
                  >
                    <Link href="/chat">
                      <BotMessageSquare className="mr-2 h-5 w-5 text-green-700" />
                      Explore Platform
                    </Link>
                  </Button>
                </motion.div>

                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                ></motion.div>
              </motion.div>

              {/* Stats */}
              <motion.div
                className="grid grid-cols-3 gap-6 pt-8"
                variants={itemVariants}
              >
                {[
                  // { value: "95.8%", label: "Accuracy" },
                  { value: "9,00,000+", label: "RAG Nodes" },
                  { value: "24/7", label: "Available" },
                ].map((stat, i) => (
                  <motion.div
                    key={i}
                    className="text-center rounded-xl bg-green-50 border border-green-200 p-6 shadow-sm hover:shadow-md transition-all duration-300"
                    whileHover={{ scale: 1.1 }}
                  >
                    <div className="text-2xl font-bold text-green-600">
                      {stat.value}
                    </div>
                    <div className="text-sm text-green-700">{stat.label}</div>
                  </motion.div>
                ))}
              </motion.div>
            </motion.div>

            {/* 3D Models */}
            <motion.div
              className="relative h-[600px] lg:h-[700px]"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.5 }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-blue-500/20 rounded-full blur-3xl" />
              <DynamicBrainPointCloudModel scale={[2, 2, 2]} />
            </motion.div>
          </motion.div>
        </section>

        {/* Features Section */}
        <FeaturesSection />

        {/* Stats Section */}
        {/* <StatsSection /> */}

        {/* CTA Section */}
        <CTASection />
      </main>
    </>
  );
}

// Features Section Component
const FeaturesSection = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  const features = [
    {
      icon: Brain,
      title: "AI Diagnostics",
      description:
        "Advanced neural networks analyze symptoms and medical data for precise diagnoses.",
    },
    {
      icon: Stethoscope,
      title: "Personalized Care",
      description:
        "Tailored treatment recommendations based on individual patient profiles.",
    },
    {
      icon: Activity,
      title: "Real-time Monitoring",
      description:
        "Continuous health monitoring with predictive analytics and alerts.",
    },
  ];

  return (
    <section ref={ref} className="py-24 bg-muted/20">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h3 className="text-4xl font-bold mb-4">
            Advanced Medical Intelligence
          </h3>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Empowering healthcare professionals with cutting-edge AI technology
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: i * 0.2 }}
              whileHover={{ y: -10, scale: 1.02 }}
              className="p-8 rounded-2xl bg-card border border-border/50 hover:border-primary/30 transition-all duration-300"
            >
              <feature.icon className="h-12 w-12 text-primary mb-6" />
              <h4 className="text-xl font-semibold mb-4">{feature.title}</h4>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Stats Section Component
const StatsSection = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  const stats = [
    {
      value: "99.8%",
      label: "Diagnostic Accuracy",
      description: "Validated across 50,000+ cases",
    },
    {
      value: "10M+",
      label: "Patients Helped",
      description: "Across 40+ countries worldwide",
    },
    {
      value: "24/7",
      label: "Always Available",
      description: "Round-the-clock medical assistance",
    },
    {
      value: "50ms",
      label: "Response Time",
      description: "Lightning-fast AI analysis",
    },
  ];

  return (
    <section ref={ref} className="py-24">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-4 gap-8">
          {stats.map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={isInView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.6, delay: i * 0.1 }}
              className="text-center"
            >
              <motion.div
                className="text-5xl font-bold bg-gradient-to-r from-primary to-blue-500 bg-clip-text text-transparent mb-2"
                whileHover={{ scale: 1.1 }}
              >
                {stat.value}
              </motion.div>
              <div className="text-lg font-semibold mb-2">{stat.label}</div>
              <div className="text-sm text-muted-foreground">
                {stat.description}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// CTA Section Component
const CTASection = () => {
  return (
    <section className="py-24 bg-gradient-to-r from-primary/10 to-blue-500/10">
      <div className="container mx-auto px-4 text-center">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="max-w-3xl mx-auto"
        >
          <h3 className="text-4xl font-bold mb-6">
            Ready to Transform Healthcare?
          </h3>
          <p className="text-xl text-muted-foreground mb-8">
            Join thousands of healthcare professionals already using MediMind
          </p>
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button size="lg" className="text-lg px-12 py-6">
              Get Started Today <ArrowRight className="ml-2" />
            </Button>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};
