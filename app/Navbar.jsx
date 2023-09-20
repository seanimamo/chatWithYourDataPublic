"use client";

import React, { useEffect, useState } from "react";

import Link from "next/link";
import { sourceCodePro } from "./styles/fonts";
import HamburgerMenu from "./components/HamburgerMenu";
const Navbar = () => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // const Navbar = () => {
  return (
    <nav className="fixed z-10 top-0 bg-gray-50 text-gray-800 w-full p-4 grid grid-cols-3 items-center">
      <a href="/" className={`text-center`}>
        Chat With Your Data
      </a>
      {isClient && <HamburgerMenu />}{" "}
      {/* Render HamburgerMenu component on the client side */}
      <div className="hidden">
        <Link href="/">Home 🏡 </Link>
        {/* Projects */}
        {/* <Link href="/page-template">Page Template ©️</Link> */}
        {/* Short Tutorials */}
        {/* <Link href="/chatcompletions">Chat Completions 💬</Link> */}
        <Link href="/pdf">PDF-GPT 👨🏻‍🏫</Link>

        {/* When Time */}
        {/* DALL E */}

        {/* <Link href="/youtubeagent">YouTube Agent ▶️ </Link> */}
        {/* <Link href="/pdf">Celebrity AI 🤳🏽</Link> */}
        {/* Tutorials */}
        {/* <Link href="/agents">Agents 🕵🏼</Link> */}
        {/* New Page: */}
        {/* E.g. For a new link to  app/newpage/page.jsx */}
        {/* <Link href="/[APP_FOLDER_NAME]">New Page 📄</Link> */}
      </div>
    </nav>
  );
};

export default Navbar;
