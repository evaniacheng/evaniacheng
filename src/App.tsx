import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import type { Variants } from 'motion/react';
import { Github, Linkedin, Mail, ExternalLink, Users, User, Globe, Coffee, Tv, Package, GraduationCap, Download, Eye, MapPin } from 'lucide-react';
import { projects, categories, experience } from './data';
import type { Category } from './data';

const cuteCharacters = [
  { name: "Cat", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Cat.png" },
  { name: "Dog", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Dog%20Face.png" },
  { name: "Panda", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Panda.png" },
  { name: "Hamster", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Hamster.png" },
  { name: "Frog", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Frog.png" },
  { name: "Hatching Chick", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Hatching%20Chick.png" },
  { name: "Rabbit", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Rabbit%20Face.png" },
  { name: "Turtle", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Turtle.png" },
  { name: "Octopus", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Octopus.png" },
  { name: "Ghost", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Ghost.png" },
  { name: "Alien", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Alien%20Monster.png" },
  { name: "Dragon", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Dragon.png" },
  { name: "Penguin", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Penguin.png" },
  { name: "Bear", url: "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Bear.png" }
];

export default function App() {
  const [activeCategory, setActiveCategory] = useState<Category>('All');
  const [charIndex, setCharIndex] = useState(0);

  // Calculate graduation progress
  const startDate = new Date('2022-09-26T00:00:00').getTime();
  const endDate = new Date('2026-06-13T00:00:00').getTime();
  const today = new Date().getTime();
  const progressPercentage = Math.min(100, Math.max(0, ((today - startDate) / (endDate - startDate)) * 100));

  const filteredProjects = projects.filter(
    (project) => activeCategory === 'All' || project.category === activeCategory
  );

  const containerVariants: Variants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants: Variants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  return (
    <div className="min-h-screen font-sans selection:bg-pink-200 selection:text-pink-900 pb-12 overflow-x-hidden">
      
      {/* Floating Navigation Pill */}
      <motion.nav 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 30, delay: 0.5 }}
        className="fixed top-6 left-1/2 -translate-x-1/2 z-50 bg-white/80 backdrop-blur-xl border border-pink-100/50 shadow-sm shadow-pink-100/20 rounded-full px-6 md:px-7 lg:px-8 py-3 md:py-3.5 lg:py-4 flex items-center gap-5 md:gap-7 lg:gap-10 w-[94%] md:w-auto justify-center"
      >
        <a href="#home" className="text-base md:text-[15px] lg:text-lg font-medium text-stone-500 hover:text-pink-500 transition-colors">Home</a>
        <a href="#projects" className="text-base md:text-[15px] lg:text-lg font-medium text-stone-500 hover:text-blue-500 transition-colors">Projects</a>
        <a href="#experience" className="text-base md:text-[15px] lg:text-lg font-medium text-stone-500 hover:text-emerald-500 transition-colors">Experience</a>
        <a href="#resume" className="text-base md:text-[15px] lg:text-lg font-medium text-stone-500 hover:text-orange-500 transition-colors">Resume</a>
      </motion.nav>

      <main className="max-w-screen-2xl mx-auto px-4 md:px-10 pt-28 space-y-20" id="home">
        
        {/* Bento Box Hero Section */}
        <motion.section 
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6"
        >
          {/* Main Intro - Span 2 */}
          <motion.div variants={itemVariants} className="md:col-span-2 bg-pink-50/80 rounded-[2rem] p-8 md:p-12 border border-pink-100/50 flex flex-col justify-center relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-pink-200/40 rounded-full mix-blend-multiply filter blur-3xl -translate-y-1/2 translate-x-1/3"></div>
            <h1 className="font-serif text-4xl md:text-6xl text-stone-800 leading-tight mb-4 relative z-10">
              Hi, I'm <span className="italic text-pink-500">Evania</span>.<br/>
              I build digital experiences.
            </h1>
            <p className="text-stone-600 text-lg max-w-md relative z-10">
              A Computer Science student passionate about frontend development, mobile apps, and AI. I love turning complex problems into beautiful, intuitive interfaces.
            </p>
            <div className="flex items-center gap-2 mt-4 text-stone-500 font-medium relative z-10">
              <MapPin size={18} className="text-pink-400" />
              Bay Area, CA
            </div>
            <div className="flex gap-4 mt-8 relative z-10">
              <a href="https://github.com" target="_blank" rel="noreferrer" className="p-3 bg-white rounded-full text-stone-600 hover:text-pink-500 hover:shadow-md transition-all">
                <Github size={20} />
              </a>
              <a href="https://linkedin.com/in/evania-cheng" target="_blank" rel="noreferrer" className="p-3 bg-white rounded-full text-stone-600 hover:text-blue-500 hover:shadow-md transition-all">
                <Linkedin size={20} />
              </a>
              <a href="mailto:evania@ucsb.edu" className="p-3 bg-white rounded-full text-stone-600 hover:text-emerald-500 hover:shadow-md transition-all">
                <Mail size={20} />
              </a>
            </div>
          </motion.div>

          {/* Education Card */}
          <motion.div variants={itemVariants} className="bg-blue-50/80 rounded-[2rem] p-8 border border-blue-100/50 flex flex-col justify-between relative overflow-hidden">
            <div className="absolute bottom-0 left-0 w-40 h-40 bg-blue-200/40 rounded-full mix-blend-multiply filter blur-2xl translate-y-1/3 -translate-x-1/3"></div>
            <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center text-blue-500 mb-6 shadow-sm relative z-10">
              <GraduationCap size={24} />
            </div>
            <div className="relative z-10 flex-1 flex flex-col">
              <h3 className="font-serif text-2xl text-stone-800 mb-2">UCSB</h3>
              <p className="text-stone-600">B.S. Computer Science</p>
              <p className="text-blue-500 font-medium mt-1">Class of 2026</p>
              
              <div className="mt-auto pt-6">
                <div className="flex justify-between text-xs font-medium text-blue-500/80 mb-1.5">
                  <span>Progress</span>
                  <span>{Math.round(progressPercentage)}%</span>
                </div>
                <div className="w-full bg-blue-200/50 rounded-full h-1.5 overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${progressPercentage}%` }}
                    transition={{ duration: 1.5, ease: "easeOut", delay: 0.5 }}
                    className="bg-blue-400 h-1.5 rounded-full"
                  />
                </div>
              </div>
            </div>
          </motion.div>

          {/* Hobbies Card */}
          <motion.div variants={itemVariants} className="bg-emerald-50/80 rounded-[2rem] p-8 border border-emerald-100/50 flex flex-col justify-center relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-200/40 rounded-full mix-blend-multiply filter blur-2xl -translate-y-1/4 translate-x-1/4"></div>
            <h3 className="font-serif text-xl text-stone-800 mb-4 relative z-10">Off the screen</h3>
            <ul className="space-y-3 relative z-10">
              <li className="flex items-center gap-3 text-stone-600"><Coffee size={18} className="text-emerald-500"/> Matcha & Cafe Hopping</li>
              <li className="flex items-center gap-3 text-stone-600"><Tv size={18} className="text-emerald-500"/> K-Dramas & C-Dramas</li>
              <li className="flex items-center gap-3 text-stone-600"><Package size={18} className="text-emerald-500"/> Blind Box Collecting</li>
            </ul>
          </motion.div>

          {/* Languages Card */}
          <motion.div variants={itemVariants} className="bg-orange-50/80 rounded-[2rem] p-8 border border-orange-100/50 flex flex-col justify-center relative overflow-hidden">
            <div className="absolute bottom-0 right-0 w-32 h-32 bg-orange-200/40 rounded-full mix-blend-multiply filter blur-2xl translate-y-1/4 translate-x-1/4"></div>
            <div className="flex items-center gap-3 mb-5 relative z-10">
              <Globe size={24} className="text-orange-400" />
              <h3 className="font-serif text-xl text-stone-800">Languages</h3>
            </div>
            <ul className="space-y-3 relative z-10 w-full">
              <li className="flex items-center justify-between text-stone-600 text-sm">
                <span className="font-medium">English</span>
                <span className="text-orange-500/80 font-medium bg-orange-100/50 px-2 py-0.5 rounded-md">Fluent</span>
              </li>
              <li className="flex items-center justify-between text-stone-600 text-sm">
                <span className="font-medium">Cantonese</span>
                <span className="text-orange-500/80 font-medium bg-orange-100/50 px-2 py-0.5 rounded-md">Fluent</span>
              </li>
              <li className="flex items-center justify-between text-stone-600 text-sm">
                <span className="font-medium">Mandarin</span>
                <span className="text-orange-500/80 font-medium bg-orange-100/50 px-2 py-0.5 rounded-md">Intermediate</span>
              </li>
            </ul>
          </motion.div>

          {/* Tech Stack Card */}
          <motion.div variants={itemVariants} className="md:col-span-1 bg-violet-50/80 rounded-[2rem] p-8 border border-violet-100/50 flex flex-col justify-center overflow-hidden relative">
            <h3 className="font-serif text-xl text-stone-800 mb-4">Tech Stack</h3>
            <div className="flex flex-wrap gap-2 relative z-10">
              {['Python', 'C++', 'React', 'Tailwind CSS', 'JavaScript', 'Plotly.js', 'LLMs', 'RAG', 'Figma'].map(tech => (
                <span key={tech} className="px-3 py-1 bg-white text-violet-600 text-sm rounded-full shadow-sm border border-violet-100/50">
                  {tech}
                </span>
              ))}
            </div>
          </motion.div>
        </motion.section>

        {/* Scrolling Animation Divider */}
        <div className="w-full overflow-hidden py-8 relative flex opacity-60">
          <div className="absolute left-0 top-0 w-12 md:w-24 h-full bg-gradient-to-r from-[#FAF9F6] to-transparent z-10 pointer-events-none"></div>
          <div className="absolute right-0 top-0 w-12 md:w-24 h-full bg-gradient-to-l from-[#FAF9F6] to-transparent z-10 pointer-events-none"></div>
          <div className="flex whitespace-nowrap animate-marquee items-center">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="flex items-center">
                <span className="font-serif italic text-xl md:text-2xl text-stone-400 mx-6 md:mx-8">Frontend</span>
                <span className="text-pink-300">✦</span>
                <span className="font-serif italic text-xl md:text-2xl text-stone-400 mx-6 md:mx-8">Mobile</span>
                <span className="text-blue-300">✦</span>
                <span className="font-serif italic text-xl md:text-2xl text-stone-400 mx-6 md:mx-8">UI/UX</span>
                <span className="text-emerald-300">✦</span>
                <span className="font-serif italic text-xl md:text-2xl text-stone-400 mx-6 md:mx-8">AI</span>
                <span className="text-violet-300">✦</span>
              </div>
            ))}
          </div>
        </div>

        {/* Projects Section */}
        <section id="projects" className="scroll-mt-32 space-y-10">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
            <h2 className="font-serif text-4xl text-stone-800">Selected Works</h2>
            
            {/* Filter Toggle */}
            <div className="flex items-center gap-2 overflow-x-auto pb-2 md:pb-0 hide-scrollbar w-full md:w-auto">
              {categories.map((category) => (
                <button
                  key={category}
                  onClick={() => setActiveCategory(category)}
                  className={`px-5 py-2 rounded-full text-sm transition-all duration-300 whitespace-nowrap ${
                    activeCategory === category 
                      ? 'bg-stone-800 text-white shadow-md' 
                      : 'bg-white text-stone-500 border border-stone-200 hover:border-stone-300 hover:text-stone-800'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>

          {/* Projects Grid */}
          <motion.div layout className="grid md:grid-cols-2 gap-8">
            <AnimatePresence mode="sync">
              {filteredProjects.map((project) => (
                <motion.div
                  key={project.id}
                  layout
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.4, type: "spring", bounce: 0.2 }}
                  className="group flex flex-col bg-white rounded-[2rem] overflow-hidden shadow-sm border border-stone-100 hover:shadow-xl hover:shadow-pink-100/50 transition-all duration-500"
                >
                  <div className="aspect-[4/3] overflow-hidden bg-stone-100 relative">
                    <img 
                      src={project.image} 
                      alt={project.title}
                      className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
                      referrerPolicy="no-referrer"
                    />
                    {/* Badges overlay */}
                    <div className="absolute top-4 left-4 flex flex-col gap-2">
                      <span className="inline-flex items-center px-3 py-1 rounded-full bg-white/90 backdrop-blur-sm text-xs font-medium text-stone-600 shadow-sm">
                        {project.category}
                      </span>
                    </div>
                    <div className="absolute top-4 right-4">
                      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full backdrop-blur-sm text-xs font-medium shadow-sm ${
                        project.projectType === 'Personal' 
                          ? 'bg-blue-50/90 text-blue-600 border border-blue-100/50' 
                          : 'bg-violet-50/90 text-violet-600 border border-violet-100/50'
                      }`}>
                        {project.projectType === 'Personal' ? <User size={12} /> : <Users size={12} />}
                        {project.projectType}
                      </span>
                    </div>
                  </div>
                  <div className="p-8 flex flex-col flex-grow">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-serif text-2xl text-stone-800">{project.title}</h3>
                      <div className="flex items-center gap-3 text-stone-400">
                        {project.github && (
                          <a href={project.github} className="hover:text-pink-500 transition-colors">
                            <Github size={20} />
                          </a>
                        )}
                        {project.link && (
                          <a href={project.link} className="hover:text-blue-500 transition-colors">
                            <ExternalLink size={20} />
                          </a>
                        )}
                      </div>
                    </div>
                    <p className="text-stone-500 leading-relaxed mb-6 flex-grow">
                      {project.description}
                    </p>
                    <div className="flex flex-wrap gap-2 mt-auto">
                      {project.tags.map(tag => (
                        <span key={tag} className="text-xs font-medium text-stone-500 bg-stone-50 px-3 py-1.5 rounded-full border border-stone-100">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        </section>

        {/* Experience Timeline Section */}
        <section id="experience" className="scroll-mt-32 max-w-4xl lg:max-w-5xl xl:max-w-6xl mx-auto">
          <h2 className="font-serif text-4xl md:text-[2.6rem] lg:text-5xl xl:text-6xl text-stone-800 mb-12 md:mb-12 lg:mb-14 text-center">Experience & Education</h2>
          
          <div className="relative border-l-2 border-stone-200 ml-3 md:ml-7 lg:ml-8 xl:ml-10 pl-8 md:pl-12 lg:pl-14 xl:pl-16 space-y-12 md:space-y-12 lg:space-y-14">
            {experience.map((item, index) => (
              <motion.div 
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="relative"
              >
                {/* Timeline dot */}
                <div className={`absolute -left-[41px] md:-left-[57px] lg:-left-[68px] xl:-left-[76px] top-1.5 w-4 h-4 lg:w-5 lg:h-5 rounded-full shadow-sm ${item.role === 'B.S. Computer Science' ? 'bg-pink-500 border-4 border-pink-200' : 'bg-[#FAF9F6] border-4 border-stone-300'}`}></div>
                
                <div className="mb-3 flex flex-col gap-2 md:grid md:grid-cols-[minmax(0,1fr)_auto] md:items-start md:gap-x-6 md:gap-y-1">
                  <div className="min-w-0">
                    <h3 className="text-xl md:text-[1.4rem] lg:text-2xl xl:text-3xl font-bold text-stone-800">{item.role}</h3>
                    <span className="text-base md:text-[1.05rem] lg:text-lg font-medium text-pink-500">{item.company}</span>
                  </div>
                  <span className="text-sm font-mono text-stone-400 bg-stone-100 px-2.5 py-1.5 rounded-md w-fit md:self-start">{item.date}</span>
                </div>
                <p className="text-stone-600 text-base md:text-[1.02rem] lg:text-lg leading-relaxed md:leading-relaxed lg:leading-8">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Resume Download Section */}
        <section id="resume" className="scroll-mt-32">
          <div className="bg-rose-50/80 border border-rose-100/50 rounded-[2rem] p-8 md:p-12 flex flex-col md:flex-row items-center justify-between gap-8 relative overflow-hidden shadow-sm">
            {/* Decorative circles */}
            <div className="absolute -top-24 -right-24 w-64 h-64 bg-rose-200/40 rounded-full mix-blend-multiply filter blur-2xl"></div>
            <div className="absolute -bottom-24 -left-24 w-64 h-64 bg-pink-200/40 rounded-full mix-blend-multiply filter blur-2xl"></div>
            
            <div className="space-y-4 max-w-xl text-center md:text-left relative z-10">
              <h2 className="font-serif text-3xl text-stone-800">Looking for a detailed breakdown?</h2>
              <p className="text-stone-600 text-lg">
                Check out my full resume for a complete history of my education, experience, and technical skills.
              </p>
            </div>
            <div className="flex flex-col lg:flex-row gap-4 w-full md:w-auto relative z-10">
              <a 
                href="/evania-cheng-resume.pdf"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 px-8 py-4 bg-white text-stone-800 rounded-full font-medium hover:bg-stone-50 hover:scale-105 shadow-sm border border-stone-200 transition-all whitespace-nowrap w-full lg:w-auto"
              >
                <Eye size={20} className="text-stone-600" />
                Preview Resume
              </a>
              <a 
                href="/evania-cheng-resume.pdf"
                download="Evania-Cheng-Resume.pdf"
                className="flex items-center justify-center gap-2 px-8 py-4 bg-stone-800 text-white rounded-full font-medium hover:bg-stone-700 hover:scale-105 shadow-md transition-all whitespace-nowrap w-full lg:w-auto"
              >
                <Download size={20} className="text-white" />
                Download Resume
              </a>
            </div>
          </div>
        </section>

      </main>

      {/* Cute Character Peeking */}
      <motion.div 
        initial={{ y: 100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 20, delay: 1 }}
        className="fixed bottom-0 right-4 md:right-12 z-50 cursor-pointer"
        onClick={() => setCharIndex((prev) => (prev + 1) % cuteCharacters.length)}
        title="Click to change character!"
      >
        <div className="relative group">
          <div className="absolute -top-10 -left-6 bg-white px-3 py-1.5 rounded-2xl rounded-br-none shadow-sm border border-stone-100 text-xs font-medium text-stone-600 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
            Click to change! ✨
          </div>
          <motion.img 
            key={cuteCharacters[charIndex].url}
            src={cuteCharacters[charIndex].url} 
            alt={cuteCharacters[charIndex].name} 
            className="w-20 h-20 md:w-24 md:h-24 object-contain drop-shadow-xl"
            initial={{ scale: 0.5, opacity: 0, rotate: -20, y: 20 }}
            animate={{ scale: 1, opacity: 1, rotate: 0, y: 0 }}
            whileHover={{ 
              scale: 1.15, 
              rotate: [0, -10, 10, -10, 0],
              y: -10,
              transition: { duration: 0.4 }
            }}
            whileTap={{ scale: 0.8, rotate: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 15 }}
          />
        </div>
      </motion.div>

      {/* Footer */}
      <footer className="border-t border-stone-200 mt-20">
        <div className="max-w-screen-2xl mx-auto px-6 md:px-10 py-8 mb-4 md:mb-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-stone-400 text-sm">
            © {new Date().getFullYear()} Evania Cheng. Built with React & Tailwind.
          </p>
          <div className="flex items-center gap-6">
            <a href="mailto:evania@ucsb.edu" className="text-stone-400 hover:text-emerald-500 transition-colors text-sm font-medium flex items-center gap-2">
              <Mail size={16} />
              evania@ucsb.edu
            </a>
            <div className="flex items-center gap-4 border-l border-stone-200 pl-6">
              <a href="https://github.com" target="_blank" rel="noreferrer" className="text-stone-400 hover:text-pink-500 transition-colors">
                <Github size={18} />
              </a>
              <a href="https://linkedin.com/in/evania-cheng" target="_blank" rel="noreferrer" className="text-stone-400 hover:text-blue-500 transition-colors">
                <Linkedin size={18} />
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
