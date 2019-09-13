# -*- mode: ruby -*-
# vi: set ft=ruby :

$bootstrap = <<END
#!/bin/bash --login
ln -sf /vagrant ~/geomesh
cd ~/geomesh
git lfs install
python -m venv .vagrant_env
source .vagrant_env/bin/activate
./setup.py install_deps
./setup.py develop
echo "source ~/geomesh/.vagrant_env/bin/activate" >> ~/.zshrc
echo "cd ~/geomesh" >> ~/.zshrc
END

Vagrant.configure("2") do |config|
  config.vm.box = "archlinux/archlinux"
  config.vm.provider "virtualbox" do |v|
    v.name = "geomesh"
    v.memory = 16384
  end
  config.ssh.forward_x11 = true
  if File.directory?(File.expand_path("~/postSandyDEM"))
    config.vm.synced_folder "~/postSandyDEM", "/vagrant_data"
    config.vm.provision :shell, privileged: false,
    inline: "ln -sf /vagrant_data /home/vagrant/postSandyDEM"
  end
  config.vm.provision :shell, privileged: true,
    inline: "pacman -Syu git git-lfs zsh grml-zsh-config base-devel cmake python python-setuptools tk gdal systemd-swap xorg --noconfirm"
  config.vm.provision :shell, privileged: true,
    inline: "systemctl enable --now systemd-swap"
  config.vm.provision :shell, privileged: true,
    inline: "chsh -s /bin/zsh vagrant"
  config.vm.provision :shell, privileged: true,
    inline: "sed -i 's/#X11Forwarding no/X11Forwarding yes/g' /etc/ssh/sshd_config"
  config.vm.provision :shell, privileged: true, inline: "systemctl restart sshd"
  config.vm.provision :shell, privileged: false,
    inline: "echo \"" << $bootstrap << "\" >> /tmp/bootstrap.sh"
  config.vm.provision :shell, privileged: false, inline: "/bin/bash --login /tmp/bootstrap.sh"
end
