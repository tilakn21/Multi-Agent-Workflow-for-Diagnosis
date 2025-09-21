"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  Environment,
  ContactShadows,
  useGLTF,
} from "@react-three/drei";
import { useRef, useMemo, useState } from "react";
import * as THREE from "three";
import dynamic from "next/dynamic";
import React from "react";

// Brain Model Component with proper TypeScript typing
function BrainMesh({
  mousePosition,
  scrollProgress,
}: {
  mousePosition: { x: number; y: number };
  scrollProgress: any;
}) {
  // ✅ Proper TypeScript typing for Three.js Mesh
  const brainRef = useRef<THREE.Mesh>(null!);
  const [hovered, setHovered] = useState(false);

  // Create brain geometry programmatically
  const brainGeometry = useMemo(() => {
    const geometry = new THREE.SphereGeometry(2, 64, 64);
    const positions = geometry.attributes.position.array;

    // Add brain-like surface irregularities
    for (let i = 0; i < positions.length; i += 3) {
      const vertex = new THREE.Vector3(
        positions[i],
        positions[i + 1],
        positions[i + 2]
      );
      const noise =
        Math.sin(vertex.x * 3) *
        Math.cos(vertex.y * 3) *
        Math.sin(vertex.z * 3) *
        0.2;
      vertex.multiplyScalar(1 + noise);
      positions[i] = vertex.x;
      positions[i + 1] = vertex.y;
      positions[i + 2] = vertex.z;
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
    return geometry;
  }, []);

  useFrame((state) => {
    if (brainRef.current) {
      // ✅ Now TypeScript knows these properties exist
      brainRef.current.rotation.x = mousePosition.y * 0.3;
      brainRef.current.rotation.y =
        mousePosition.x * 0.3 + state.clock.elapsedTime * 0.2;

      // Gentle floating animation
      brainRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.3;

      // Scale based on scroll progress
      const scrollScale = 1 + (scrollProgress?.get() || 0) * 0.5;
      brainRef.current.scale.setScalar(scrollScale);

      // Hover effect - properly typed material
      if (brainRef.current.material instanceof THREE.MeshStandardMaterial) {
        brainRef.current.material.emissive.setHSL(
          0.6,
          hovered ? 0.4 : 0.2,
          hovered ? 0.3 : 0.1
        );
      }
    }
  });

  return (
    <group>
      <mesh
        ref={brainRef}
        geometry={brainGeometry}
        onPointerEnter={() => setHovered(true)}
        onPointerLeave={() => setHovered(false)}
        castShadow
        receiveShadow
      >
        <meshStandardMaterial
          color="#ff6b6b"
          roughness={0.3}
          metalness={0.8}
          emissive="#ff6b6b"
          emissiveIntensity={0.1}
          transparent
          opacity={0.9}
        />
      </mesh>

      {/* Neural network visualization */}
      <NeuralConnections />
    </group>
  );
}

// Neural Connections Component with proper typing
function NeuralConnections() {
  const connectionsRef = useRef<THREE.Group>(null!);

  const connections = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const connections: [THREE.Vector3, THREE.Vector3][] = [];

    // Generate random points around the brain
    for (let i = 0; i < 50; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      const radius = 2.5 + Math.random() * 1;

      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.sin(phi) * Math.sin(theta);
      const z = radius * Math.cos(phi);

      points.push(new THREE.Vector3(x, y, z));
    }

    // Create connections between nearby points
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        if (points[i].distanceTo(points[j]) < 2) {
          connections.push([points[i], points[j]]);
        }
      }
    }

    return connections;
  }, []);

  useFrame((state) => {
    if (connectionsRef.current) {
      connectionsRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <group ref={connectionsRef}>
      {connections.map((connection, i) => (
        <line key={i}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes.position"
              args={[
                new Float32Array([
                  connection[0].x,
                  connection[0].y,
                  connection[0].z,
                  connection[1].x,
                  connection[1].y,
                  connection[1].z,
                ]),
                3, // itemSize
              ]}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#4ade80" opacity={0.3} transparent />
        </line>
      ))}
    </group>
  );
}

// Defined the BrainPointCloudModelProps interface to fix the missing type error
interface BrainPointCloudModelProps {
  scale?: [number, number, number];
}

// Increased the scale of the BrainPointCloudModel to make it larger
export function BrainPointCloudModel({
  scale = [100, 100, 100], // Increased the scale for the model
}: BrainPointCloudModelProps) {
  const { scene } = useGLTF("/models/brain_hologram.glb"); // Updated model path

  // Apply green material and hologram effect to the model
  scene.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      (child as THREE.Mesh).material = new THREE.MeshStandardMaterial({
        color: "#4ade80", // Green color
        roughness: 0.1,
        metalness: 0.8,
        emissive: "#4ade80",
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.8,
      });
    }
  });

  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [0, 0, 8], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        className="cursor-grab active:cursor-grabbing"
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
        <pointLight position={[-10, -10, -5]} intensity={0.5} color="#4ade80" />

        {/* Add glowing outline effect */}
        <primitive object={scene} scale={scale} />
        <primitive
          object={scene.clone()}
          scale={scale.map((s) => s * 1.05)}
          renderOrder={1}
          onBeforeRender={(
            renderer: THREE.WebGLRenderer,
            scene: THREE.Scene,
            camera: THREE.Camera,
            geometry: THREE.BufferGeometry,
            material: THREE.Material
          ) => {
            (material as THREE.MeshBasicMaterial).color.set("#4ade80");
            (material as THREE.MeshBasicMaterial).opacity = 0.5;
          }}
        />

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotateSpeed={1}
          maxPolarAngle={Math.PI / 1.5}
          minPolarAngle={Math.PI / 3}
        />
      </Canvas>
    </div>
  );
}

// Wrap the main 3D Canvas in a separate component for dynamic import
function Brain3DCanvas({
  mousePosition,
  scrollProgress,
}: {
  mousePosition: { x: number; y: number };
  scrollProgress: any;
}) {
  return (
    <Canvas
      camera={{ position: [0, 0, 8], fov: 50 }}
      gl={{ antialias: true, alpha: true }}
      className="cursor-grab active:cursor-grabbing"
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
      <pointLight position={[-10, -10, -5]} intensity={0.5} color="#4ade80" />
      <BrainMesh
        mousePosition={mousePosition}
        scrollProgress={scrollProgress}
      />
      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotateSpeed={1}
        maxPolarAngle={Math.PI / 1.5}
        minPolarAngle={Math.PI / 3}
      />
    </Canvas>
  );
}

// Dynamically import the Brain3DCanvas component
const DynamicBrain3DCanvas = dynamic(() => Promise.resolve(Brain3DCanvas), {
  ssr: false,
});

// Non-SSR Wrapper Component
const NonSSRWrapper = (props: { children: React.ReactNode }) => (
  <React.Fragment>{props.children}</React.Fragment>
);

const DynamicNonSSRWrapper = dynamic(() => Promise.resolve(NonSSRWrapper), {
  ssr: false,
});

// Only export the dynamic version for client-side rendering
const DynamicBrainPointCloudModel = dynamic(
  () => Promise.resolve(BrainPointCloudModel),
  { ssr: false }
);

export { DynamicBrainPointCloudModel };

// Exported component using dynamic import
function BrainModel3D({
  mousePosition,
  scrollProgress,
}: {
  mousePosition: { x: number; y: number };
  scrollProgress: any;
}) {
  return (
    <DynamicBrain3DCanvas
      mousePosition={mousePosition}
      scrollProgress={scrollProgress}
    />
  );
}

// Exported component using NonSSRWrapper for client-only rendering
const BrainModel3DClientOnly = (props: {
  mousePosition: { x: number; y: number };
  scrollProgress: any;
}) => (
  <DynamicNonSSRWrapper>
    <BrainModel3D {...props} />
  </DynamicNonSSRWrapper>
);

export default BrainModel3DClientOnly;
