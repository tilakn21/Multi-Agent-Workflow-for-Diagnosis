'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface BodyModelProps {
  onSelect: (part: string) => void;
}

export function BodyModel({ onSelect }: BodyModelProps) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const currentMount = mountRef.current;
    
    // Scene
    const scene = new THREE.Scene();
    
    // Camera
    const camera = new THREE.PerspectiveCamera(50, currentMount.clientWidth / currentMount.clientHeight, 0.1, 1000);
    camera.position.z = 20;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    currentMount.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 10;
    controls.maxDistance = 50;
    controls.enablePan = false;
    
    // Materials
    const wireframeMaterial = new THREE.MeshBasicMaterial({ color: 0x2EC1B1, wireframe: true });
    const highlightMaterial = new THREE.MeshBasicMaterial({ color: 0x2EC1B1, transparent: true, opacity: 0.5 });
    
    const bodyParts: THREE.Mesh[] = [];

    // Body parts
    const createPart = (geometry: THREE.BufferGeometry, name: string, position: [number, number, number]) => {
      const part = new THREE.Mesh(geometry, wireframeMaterial);
      part.name = name;
      part.position.set(...position);
      scene.add(part);
      bodyParts.push(part);
      return part;
    };

    const head = createPart(new THREE.SphereGeometry(1.5, 16, 16), 'Head', [0, 7, 0]);
    const torso = createPart(new THREE.CylinderGeometry(2, 2, 4, 16), 'Abdomen', [0, 3.5, 0]);
    const leftArm = createPart(new THREE.CylinderGeometry(0.5, 0.5, 3, 16), 'Left Arm', [-3, 4, 0]);
    const rightArm = createPart(new THREE.CylinderGeometry(0.5, 0.5, 3, 16), 'Right Arm', [3, 4, 0]);
    const leftLeg = createPart(new THREE.CylinderGeometry(0.7, 0.7, 4, 16), 'Left Leg', [-1.2, -0.5, 0]);
    const rightLeg = createPart(new THREE.CylinderGeometry(0.7, 0.7, 4, 16), 'Right Leg', [1.2, -0.5, 0]);

    // Raycaster for clicks
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let selectedPart: THREE.Mesh | null = null;

    const onMouseClick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(bodyParts);

      if (intersects.length > 0) {
        if (selectedPart) {
          selectedPart.material = wireframeMaterial;
        }
        selectedPart = intersects[0].object as THREE.Mesh;
        selectedPart.material = highlightMaterial;
        onSelect(selectedPart.name);
      }
    };
    
    currentMount.addEventListener('click', onMouseClick);
    
    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
        if (!currentMount) return;
        camera.aspect = currentMount.clientWidth / currentMount.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      currentMount.removeEventListener('click', onMouseClick);
      if (renderer.domElement.parentNode === currentMount) {
        currentMount.removeChild(renderer.domElement);
      }
    };
  }, [onSelect]);

  return <div ref={mountRef} className="w-full h-64 md:h-96" />;
}
